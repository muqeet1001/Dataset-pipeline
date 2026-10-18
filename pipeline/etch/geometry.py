"""Visible surface observations and supervised cloth-to-body targets, in metres."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh


def topology_digest(faces, vertex_count):
    values=np.asarray(faces,dtype='<i8')
    return hashlib.sha256(str(vertex_count).encode()+values.tobytes()).hexdigest()


def surface_sample(mesh, count, rng):
    if count<1 or not np.isfinite(mesh.area_faces).all() or mesh.area<=0:
        raise ValueError('Need a finite nonempty triangle surface and positive sample count')
    face=rng.choice(len(mesh.faces),count,p=mesh.area_faces/mesh.area)
    uv=rng.random((count,2));s=np.sqrt(uv[:,0])
    bary=np.column_stack([1-s,s*(1-uv[:,1]),s*uv[:,1]])
    return np.einsum('ni,nij->nj',bary,mesh.triangles[face]),face,bary


def camera_positions(mesh):
    directions=np.asarray([[x,y,z] for x in (-1,0,1) for y in (-1,0,1)
                           for z in (-1,0,1) if x or y or z],dtype=float)
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    return mesh.bounds.mean(0)+directions*np.linalg.norm(mesh.extents)*2.5


def visible_points(mesh, points, normals=None, cameras=None):
    """Test real line-of-sight against all scene surfaces, including occluders.

    Full body + garments are not themselves a scan: body surfaces behind cloth
    must be rejected. Any of 26 fixed external cameras may observe a point.
    """
    points=np.asarray(points,dtype=float)
    cameras=camera_positions(mesh) if cameras is None else np.asarray(cameras)
    visible=np.zeros(len(points),dtype=bool)
    epsilon=max(float(np.linalg.norm(mesh.extents))*1e-6,1e-7)
    for camera in cameras:
        pending=np.flatnonzero(~visible)
        if not len(pending):break
        for start in range(0,len(pending),256):
            ids=pending[start:start+256]
            delta=camera-points[ids];distance=np.linalg.norm(delta,axis=1)
            direction=delta/np.maximum(distance[:,None],1e-12)
            if normals is not None:
                front=np.einsum('ij,ij->i',np.asarray(normals)[ids],direction)>1e-8
                ids=ids[front];direction=direction[front];distance=distance[front]
            if not len(ids):continue
            origins=points[ids]+direction*epsilon
            locations,rays,_=mesh.ray.intersects_location(origins,direction,multiple_hits=False)
            blocked=np.zeros(len(ids),dtype=bool)
            if len(rays):
                hit_distance=np.linalg.norm(locations-origins[rays],axis=1)
                blocked[rays[hit_distance<distance[rays]-epsilon]]=True
            visible[ids[~blocked]]=True
    return visible


def visible_surface_sample(mesh, count, seed=1):
    rng=np.random.default_rng(seed);points=[];faces=[];barys=[];remaining=count
    tested=0
    for _ in range(12):
        if remaining<=0:break
        p,f,b=surface_sample(mesh,max(remaining*2,128),rng);tested+=len(p)
        keep=visible_points(mesh,p,mesh.face_normals[f])
        p,f,b=p[keep][:remaining],f[keep][:remaining],b[keep][:remaining]
        points.extend(p);faces.extend(f);barys.extend(b);remaining-=len(p)
    if remaining:raise ValueError('Could not obtain enough externally visible surface samples')
    return np.asarray(points),np.asarray(faces),np.asarray(barys),tested


def heat_distances(body, indices):
    import potpourri3d as pp3d
    solver=pp3d.MeshHeatMethodDistanceSolver(np.asarray(body.vertices),np.asarray(body.faces))
    distances=np.asarray([solver.compute_distance(int(i)) for i in indices])
    if not np.isfinite(distances).all():raise ValueError('Body geodesic distances are not finite')
    return np.maximum(distances,0)


def create_markers(body, count=86):
    """Choose stable vertex IDs ONCE, using geodesic farthest-point coverage.

    These are MHR markers, not anatomical SMPL marker correspondences. Reuse
    this exact definition for every mesh with the same MHR topology.
    """
    import potpourri3d as pp3d
    if count<4 or count>len(body.vertices):raise ValueError('Invalid marker count')
    solver=pp3d.MeshHeatMethodDistanceSolver(np.asarray(body.vertices),np.asarray(body.faces))
    selected=[];nearest=np.full(len(body.vertices),np.inf)
    index=int(np.argmax(body.vertices[:,1]))
    for _ in range(count):
        selected.append(index)
        distance=np.asarray(solver.compute_distance(index))
        if not np.isfinite(distance).all():raise ValueError('Markers require a connected body surface')
        nearest=np.minimum(nearest,np.maximum(distance,0));nearest[selected]=-1
        index=int(nearest.argmax())
    return {'body_model':'MHR','count':count,'vertex_indices':selected,
        'labels':[f'mhr_{i:02d}' for i in range(count)],
        'topology_sha256':topology_digest(body.faces,len(body.vertices)),
        'selection':'geodesic farthest-point coverage on supplied male reference, Y-up',
        'anatomical_smpl_correspondence':False}


def closest_body(body, points):
    closest=[];faces=[]
    for start in range(0,len(points),256):
        p,_,f=trimesh.proximity.closest_point(body,points[start:start+256])
        closest.extend(p);faces.extend(f)
    closest=np.asarray(closest);faces=np.asarray(faces,dtype=int)
    bary=trimesh.triangles.points_to_barycentric(body.triangles[faces],closest)
    return closest,faces,bary


def normal_anchors(body, clothes, count=6000, seed=2, max_distance=.16):
    """Body-normal rays with the reverse body-intersection check in ETCH.

    Select the furthest garment layer within range, then require its backward
    ray to return to the original body point. Visibility is checked separately
    against the complete scene. Skin is handled by the nearest-body fallback.
    """
    p,face,bary=surface_sample(body,count,np.random.default_rng(seed))
    normals=np.einsum('ni,nij->nj',bary,body.vertex_normals[body.faces[face]])
    normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
    outer=[];inner=[]
    for start in range(0,count,256):
        pp=p[start:start+256];nn=normals[start:start+256]
        hits,rays,_=clothes.ray.intersects_location(pp+nn*1e-6,nn,multiple_hits=True)
        if not len(rays):continue
        distance=np.linalg.norm(hits-pp[rays],axis=1)
        chosen={}
        for j in np.argsort(distance):
            if 1e-6<distance[j]<max_distance:chosen[int(rays[j])]=int(j)
        if not chosen:continue
        rr=np.asarray(list(chosen));xx=hits[list(chosen.values())]
        back,back_rays,_=body.ray.intersects_location(xx-nn[rr]*1e-6,-nn[rr],multiple_hits=False)
        valid=np.linalg.norm(back-pp[rr[back_rays]],axis=1)<1e-4
        outer.extend(xx[back_rays[valid]]);inner.extend(pp[rr[back_rays[valid]]])
    return np.asarray(outer).reshape(-1,3),np.asarray(inner).reshape(-1,3)


def pair_targets(body, scene, garments, markers, count=5000, seed=1, anchor_count=6000):
    if topology_digest(body.faces,len(body.vertices))!=markers['topology_sha256']:
        raise ValueError('Marker definition and body topology differ')
    points,outer_face,_,tested=visible_surface_sample(scene,count,seed)
    inner,body_face,body_bary=closest_body(body,points)
    anchors,anchor_body=normal_anchors(body,garments,anchor_count,seed+1)
    anchor_transfer=np.zeros(count,dtype=bool)
    if len(anchors):
        visible=visible_points(scene,anchors)
        anchors,anchor_body=anchors[visible],anchor_body[visible]
        if len(anchors):
            distance,index=cKDTree(anchors).query(points)
            anchor_transfer=distance<.01
            # Match the released implementation: transfer the displacement,
            # rather than accidentally swapping the vector convention.
            inner[anchor_transfer]=points[anchor_transfer]+(
                anchor_body[index[anchor_transfer]]-anchors[index[anchor_transfer]])
            _,body_face,body_bary=closest_body(body,inner)
    vectors=inner-points;magnitude=np.linalg.norm(vectors,axis=1)
    direction=vectors/np.maximum(magnitude[:,None],1e-12)
    indices=np.asarray(markers['vertex_indices'],dtype=int)
    geodesic=heat_distances(body,indices)
    distances=np.einsum('kni,ni->kn',geodesic[:,body.faces[body_face]],body_bary)
    labels=distances.argmin(0);nearest=distances[labels,np.arange(count)]
    center=scene.bounds.mean(0)
    arrays={'outer_points':(points-center).astype('f4'),
        'inner_points':(inner-center).astype('f4'),'vectors_inward':vectors.astype('f4'),
        'magnitude_m':magnitude.astype('f4'),'direction_inward':direction.astype('f4'),
        'direction_valid':magnitude>1e-6,'marker_labels':labels.astype('i8'),
        'marker_confidence':np.exp(-10*np.maximum(nearest,0)).astype('f4'),
        'markers':(body.vertices[indices]-center).astype('f4'),
        'marker_vertex_indices':indices,'center_m':center.astype('f8'),
        'outer_face':outer_face,'anchor_transfer':anchor_transfer}
    facts={'points':count,'candidate_points_tested':tested,'normal_anchors':len(anchors),
        'anchor_transferred_points':int(anchor_transfer.sum()),'visible_marker_labels':len(np.unique(labels)),
        'magnitude_p95_m':float(np.percentile(magnitude,95)),
        'vector_convention':'inner_body_minus_outer_clothing; inner = outer + vector',
        'anchor_matching':'released-code Euclidean nearest anchor within 0.01 m; paper describes geodesic nearest',
        'marker_distance':'heat-method geodesic on native MHR surface, barycentric interpolation',
        'marker_scope':'86 native MHR surface markers; not SMPL marker IDs',
        'observation_scope':'uniform visible surface samples from 26 external viewpoints; not RGB video reconstruction'}
    return arrays,facts


def load_pair(path, require_training_ready=True):
    path=Path(path);receipt=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if require_training_ready and not receipt.get('training_ready',False):
        raise ValueError('Pair is development-only: upstream clothing or data checks have not passed')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['npz_sha256']:
        raise ValueError('Pair checksum differs from receipt')
    with np.load(path,allow_pickle=False) as values:
        arrays={key:values[key].copy() for key in values.files}
    expected=arrays['outer_points']+arrays['vectors_inward']
    if not np.isfinite(expected).all() or not np.allclose(expected,arrays['inner_points'],atol=2e-7):
        raise ValueError('Pair vector convention or finite values are invalid')
    return arrays,receipt
