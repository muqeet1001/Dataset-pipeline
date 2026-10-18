"""Independent geometry checks; no training-ready claim from appearance alone."""
import numpy as np
import trimesh


def edge_crossings(source, target, self_test=False):
    edges = source.edges_unique
    origins = source.vertices[edges[:, 0]]
    delta = source.vertices[edges[:, 1]] - origins
    lengths = np.linalg.norm(delta, axis=1)
    directions = delta / lengths[:, None]
    index_tri, index_ray, locations = target.ray.intersects_id(
        ray_origins=origins, ray_directions=directions, multiple_hits=True, return_locations=True)
    if len(index_ray) == 0:
        return 0
    distances = np.einsum('ij,ij->i', locations - origins[index_ray], directions[index_ray])
    keep = (distances > 1e-5) & (distances < lengths[index_ray] - 1e-5)
    if self_test:
        incident = (target.faces[index_tri, :, None] == edges[index_ray, None, :]).any(axis=(1, 2))
        keep &= ~incident
    return int(len(np.unique(index_ray[keep])))


def validate(body, garments, settled, history):
    result = {'finite': True, 'settled': bool(settled), 'garments': {},
              'tolerances': {'body_penetration_m': .001, 'segment_endpoint_exclusion_m': 1e-5},
              'physical_fit_calibrated': False,
              'limitations': ['Original prototype tee and elastic-waist trouser sewing patterns', 'Fabric parameters are uncalibrated assumptions',
                              'No comparison against physical garments or MD/VStitcher yet']}
    for garment in garments:
        mesh = garment['mesh']
        finite = bool(np.isfinite(mesh.vertices).all())
        result['finite'] &= finite
        if not finite:
            result['garments'][garment['name']] = {'finite': False}
            continue
        points = np.vstack([mesh.vertices, mesh.triangles_center])
        signed = trimesh.proximity.signed_distance(body, points)
        edges = mesh.edges_unique
        actual_lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
        strain = actual_lengths / garment['rest_edge_lengths'] - 1
        result['garments'][garment['name']] = {
            'vertices': len(mesh.vertices), 'triangles': len(mesh.faces),
            'body_inside_samples_over_tolerance': int((signed > .001).sum()),
            'max_body_penetration_m': float(max(0, signed.max())),
            'body_crossing_edges': edge_crossings(mesh, body),
            'self_crossing_edges': edge_crossings(mesh, mesh, self_test=True),
            'edge_strain_p95_abs': float(np.percentile(np.abs(strain), 95)),
            'edge_strain_max_abs': float(np.abs(strain).max()),
        }
    a, b = [g['mesh'] for g in garments]
    result['inter_garment_crossing_edges'] = edge_crossings(a, b) + edge_crossings(b, a)
    result['settling_history'] = history[-30:]
    result['geometry_pass'] = bool(result['finite'] and all(
        g.get('body_inside_samples_over_tolerance', 1) == 0 and g.get('body_crossing_edges', 1) == 0 and
        g.get('self_crossing_edges', 1) == 0 and g.get('edge_strain_p95_abs', 1) < .15
        for g in result['garments'].values()) and result['inter_garment_crossing_edges'] == 0)
    result['numerical_pass'] = result['geometry_pass'] and result['settled']
    return result
