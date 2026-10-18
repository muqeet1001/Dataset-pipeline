"""Body-derived prototype cloth shells with explicit physical rest metrics.

This is an automatic shell draft, not a production sewing-pattern system.
It cuts neck/hem/cuff openings, grades the cloth rest surface, and then lets
Newton solve the free cloth. The source body is never modified.
"""
import numpy as np
import trimesh

PROFILES = {
    'tight': {'shirt_ease_m': .012, 'pants_ease_m': .008},
    'medium': {'shirt_ease_m': .030, 'pants_ease_m': .023},
    'loose': {'shirt_ease_m': .055, 'pants_ease_m': .045},
}


def _cut(mesh, normal, point):
    result = mesh.slice_plane(plane_origin=point, plane_normal=normal, cap=False)
    result.merge_vertices()
    result.remove_unreferenced_vertices()
    return result


def draft(body, fit, face_budget=1600):
    if fit not in PROFILES:
        raise ValueError(f'Unknown fit: {fit}')
    height = float(body.extents[1])
    # Normalized landmarks apply only to this verified, upright MHR A-pose.
    # Future avatar poses require an explicit landmark adapter.
    scale = height / 1.776
    shirt = _cut(body.copy(), [0, 1, 0], [0, 1.00 * scale, 0])
    shirt = _cut(shirt, [0, -1, 0], [0, 1.548 * scale, 0])
    for side in (-1, 1):
        shirt = _cut(shirt, [-side, .80, 0], [side * .355 * scale, 1.286 * scale, 0])
    pants = _cut(body.copy(), [0, 1, 0], [0, .135 * scale, 0])
    pants = _cut(pants, [0, -1, 0], [0, 1.025 * scale, 0])
    garments = []
    for name, shell in [('shirt', shirt), ('pants', pants)]:
        components = shell.split(only_watertight=False, repair=False)
        shell = max(components, key=lambda m: m.area)
        shell = shell.simplify_quadric_decimation(face_count=face_budget)
        shell.remove_unreferenced_vertices()
        # Collapse clipping slivers before constructing physical rest metrics.
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import coo_matrix
        for _ in range(3):
            edges = shell.edges_unique
            short = shell.edges_unique_length < .004
            if not short.any():
                break
            e = edges[short]
            graph = coo_matrix((np.ones(len(e)*2), (np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),
                               shape=(len(shell.vertices),len(shell.vertices)))
            count, labels = connected_components(graph,directed=False)
            vertices = np.zeros((count,3)); np.add.at(vertices,labels,shell.vertices)
            vertices /= np.bincount(labels)[:,None]
            faces = labels[shell.faces]
            keep = (faces[:,0]!=faces[:,1]) & (faces[:,0]!=faces[:,2]) & (faces[:,1]!=faces[:,2])
            shell = trimesh.Trimesh(vertices,faces[keep],process=True)
            shell.update_faces(shell.nondegenerate_faces(height=1e-7))
            shell.remove_unreferenced_vertices()
        # Fit the draft to the actual supplied MHR surface after decimation.
        nearest, _, faces = body.nearest.on_surface(shell.vertices)
        normals = body.face_normals[faces]
        ease = np.full(len(shell.vertices), PROFILES[fit][name + '_ease_m'] * scale)
        y = nearest[:, 1]
        if name == 'pants':
            # Same supported waistband across sizes, with graded hip/thigh/leg.
            blend = np.clip((1.025 * scale - y) / (.10 * scale), 0, 1)
            ease = .006 * scale * (1 - blend) + ease * blend
        else:
            # A consistent neck opening supports all three shirt sizes.
            neck = np.clip((y - 1.49 * scale) / (.058 * scale), 0, 1)
            ease = ease * (1 - neck) + .009 * scale * neck
        shell.vertices = nearest + normals * ease[:, None]
        if not shell.is_winding_consistent or shell.area_faces.min() < 1e-10:
            raise ValueError(f'{name} draft has invalid triangles')
        # Each triangle has an isometric chart in metres. This preserves its
        # graded rest lengths without pretending appearance UVs are patterns.
        tri = shell.triangles
        a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        length = np.linalg.norm(a, axis=1)
        x = np.einsum('ij,ij->i', a, b) / length
        z = np.sqrt(np.maximum(np.einsum('ij,ij->i', b, b) - x*x, 0))
        charts = np.stack([np.zeros((len(tri), 2)), np.column_stack([length, np.zeros(len(tri))]),
                           np.column_stack([x, z])], axis=1).reshape(-1, 2)
        pinned = np.zeros(len(shell.vertices), dtype=bool)
        if name == 'pants':
            pinned = shell.vertices[:, 1] > shell.bounds[1, 1] - .012 * scale
            if pinned.sum() < 3:
                raise ValueError('Pants waistband could not be supported')
        else:
            # Six collar seam supports stabilize placement on this static body.
            # They are explicitly recorded, not hidden whole-garment skinning.
            for side in (-1,1):
                candidates=np.flatnonzero((shell.vertices[:,1]>1.505*scale) &
                    (shell.vertices[:,0]*side>.015*scale) & (shell.vertices[:,0]*side<.10*scale))
                chosen=candidates[np.argsort(shell.vertices[candidates,1])[-3:]]
                pinned[chosen]=True
        edges = shell.edges_unique
        garments.append({'name': name, 'mesh': shell, 'panel_vertices': charts,
                         'panel_indices': np.arange(len(charts)).reshape(-1, 3),
                         'pinned': pinned,
                         'rest_edge_lengths': np.linalg.norm(shell.vertices[edges[:, 0]] - shell.vertices[edges[:, 1]], axis=1),
                         'draft': {'method': 'body-derived open shell with graded rest metric',
                                   'profile': PROFILES[fit], 'support_vertices': int(pinned.sum()),
                                   'support_type': 'waistband' if name=='pants' else 'six collar seam supports',
                                   'landmark_scope': 'upright MHR A-pose; proportions calibrated to supplied male avatar',
                                   'fabric_scope': 'assumed isotropic preset, not calibrated real fabric',
                                   'pattern_scope': 'local triangle charts, not editable tailor sewing patterns'}})
    return garments
