"""Meaningful input, grading and collision regression tests for the local pilot."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import trimesh
import warp as wp

from pipeline.avatar_clothing import load_body, ROOT
from pipeline.body_contact import BodyContact
from pipeline.cloth_quality import edge_crossings
from pipeline.garment_draft import draft


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
            areas[g['name']].append(g['mesh'].area)
            if g['name']=='shirt':
                assert 0 < g['pinned'].sum() <= 6
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
