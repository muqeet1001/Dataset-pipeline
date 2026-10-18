"""Meaningful input, grading and collision regression tests for the local pilot."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import trimesh
wp = pytest.importorskip('warp', reason='Install requirements-cloth.txt for the optional cloth tests')
pytest.importorskip('skimage', reason='Install requirements-cloth.txt for the optional cloth tests')

from pipeline.avatar_clothing import load_body, ROOT
from pipeline.body_contact import BodyContact
from pipeline.cloth_quality import edge_crossings
from pipeline.sewing_patterns import draft
from pipeline.cloth_receipts import job_lock, reusable


def test_supplied_body_hash_units_and_dimensions():
    body, metadata, sha = load_body(ROOT / 'inputs/MHR-male-10027')
    assert sha == 'd200f62501854dc6f4ed3070a97b5df4e68d3203749b056ca2bd13f6999f3fad'
    assert len(body.vertices) == 18439
    assert len(body.faces) == 36874
    assert body.extents[1] == pytest.approx(1.776, abs=1e-6)
    assert metadata['sex'] == 'male'


def test_unknown_units_rejected_before_geometry_load(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps({'obj_units':'unknown','vertical_axis':'Y'}))
    with pytest.raises(ValueError,match='explicit centimetre'):
        load_body(tmp_path)


def test_drafts_grade_cloth_and_preserve_body():
    body, _, _ = load_body(ROOT / 'inputs/MHR-male-10027')
    before = body.vertices.copy()
    areas = {'shirt':[], 'pants':[]}
    for fit in ('tight','medium','loose'):
        garments = draft(body,fit,1200)
        assert len(garments) == 2
        for g in garments:
            assert np.isfinite(g['panel_vertices']).all()
            assert np.min(g['rest_edge_lengths']) > 0
            # Sewing must leave just neck/cuffs/hem, or waist/two leg hems.
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            mesh=g['mesh']
            boundary=mesh.edges_unique[np.bincount(mesh.edges_unique_inverse)==1]
            graph=coo_matrix((np.ones(len(boundary)*2),
                (np.r_[boundary[:,0],boundary[:,1]],np.r_[boundary[:,1],boundary[:,0]])),
                shape=(len(mesh.vertices),)*2)
            _,labels=connected_components(graph,directed=False)
            openings=len(np.unique(labels[boundary.ravel()]))
            assert openings==(4 if g['name']=='shirt' else 3)
            assert np.bincount(mesh.edges_unique_inverse).max()==2
            assert mesh.is_winding_consistent
            assert mesh.area_faces.min()>1e-10
            uv=g['panel_vertices'][g['panel_indices']]
            a,b=uv[:,1]-uv[:,0],uv[:,2]-uv[:,0]
            flat_area=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0]).sum()/2
            assert flat_area==pytest.approx(g['draft']['flat_area_m2'])
            # A pattern metric must not inherit body curvature or placement stretch.
            assert np.max(np.abs(mesh.edges_unique_length/g['rest_edge_lengths']-1))>.05
            areas[g['name']].append(flat_area)
            if g['name']=='shirt':
                assert g['pinned'].sum()==0
            else:
                assert 3 <= g['pinned'].sum() < len(g['mesh'].vertices)*.2
    np.testing.assert_array_equal(before,body.vertices)
    assert all(np.diff(a).min()>0 for a in areas.values())


def test_triangle_edge_crossing_detects_collision_with_outside_endpoints():
    body = trimesh.creation.box(extents=[1,1,1])
    cloth = trimesh.Trimesh(vertices=[[-1,0,0],[1,0,0],[1,0,1]],faces=[[0,1,2]],process=False)
    assert not body.contains(cloth.vertices).any()
    assert edge_crossings(cloth,body)>0
    outside = cloth.copy(); outside.apply_translation([0,3,0])
    assert edge_crossings(outside,body)==0


def test_contact_resolves_penetration_without_moving_free_or_pinned_points():
    wp.init()
    body = trimesh.creation.box(extents=[2,2,2])
    original = body.vertices.copy()
    # First inside movable point, second outside free point, third pinned point.
    points = np.array([[.9,0,0],[1.2,0,0],[.9,.2,0]],dtype=np.float32)
    mesh = trimesh.Trimesh(vertices=points,faces=np.empty((0,3),dtype=int),process=False)
    model=SimpleNamespace(device=wp.get_device('cpu'),particle_count=3,
                          particle_inv_mass=wp.array([1.,1.,0.],dtype=float,device='cpu'))
    contacts=BodyContact(body,[{'mesh':mesh}],model)
    state=SimpleNamespace(particle_q=wp.array(points,dtype=wp.vec3,device='cpu'),
                          particle_qd=wp.zeros(3,dtype=wp.vec3,device='cpu'))
    contacts.solve(state,iterations=10)
    actual=state.particle_q.numpy()
    assert actual[0,0] >= 1.0019
    np.testing.assert_array_equal(actual[1:],points[1:])
    np.testing.assert_array_equal(body.vertices,original)


def test_corrupt_and_incomplete_outputs_are_not_reused(tmp_path):
    import hashlib
    artifact=tmp_path/'scene.glb'
    artifact.write_bytes(b'original')
    report=tmp_path/'fit-report.json'
    report.write_text(json.dumps({'signature':'same','status':'numerical_pass_unvalidated_fabric',
        'artifacts':{'scene.glb':hashlib.sha256(b'original').hexdigest()}}))
    assert reusable(report,'same') is not None
    assert reusable(report,'changed') is None
    artifact.write_bytes(b'corrupt')
    assert reusable(report,'same') is None
    artifact.unlink()
    assert reusable(report,'same') is None
    report.write_text('{partial')
    assert reusable(report,'same') is None


def test_body_contact_cannot_push_fabric_through_another_layer():
    wp.init()
    body=trimesh.creation.box(extents=[2,2,2])
    triangle=np.array([[.995,-.4,-.4],[.995,.4,-.4],[.995,0,.4]],dtype=np.float32)
    other=triangle.copy();other[:,0]=1.002
    points=np.vstack([triangle,other])
    mesh=trimesh.Trimesh(points,[[0,1,2],[3,4,5]],process=False)
    model=SimpleNamespace(device=wp.get_device('cpu'),particle_count=6,
        particle_inv_mass=wp.array([1.,1.,1.,0.,0.,0.],dtype=float,device='cpu'))
    contacts=BodyContact(body,[{'mesh':mesh}],model)
    state=SimpleNamespace(particle_q=wp.array(points,dtype=wp.vec3,device='cpu'),
        particle_qd=wp.zeros(6,dtype=wp.vec3,device='cpu'))
    contacts.solve(state,iterations=10)
    actual=state.particle_q.numpy()
    assert (actual[:3,0]>.995).all()
    assert (actual[:3,0]<1.002).all()
    np.testing.assert_array_equal(actual[3:],other)


def test_job_lock_is_exclusive_and_released_on_failure(tmp_path):
    with pytest.raises(ValueError):
        with job_lock(tmp_path):
            with pytest.raises(RuntimeError,match='Another clothing job'):
                with job_lock(tmp_path):
                    pass
            raise ValueError('test interruption')
    with job_lock(tmp_path):
        pass
