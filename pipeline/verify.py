"""Reproduce stored meshes from their exact official model inputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import trimesh
from .mhr import topology


def verify_generated(model, directory: Path):
    reports = []
    inputs = sorted(directory.glob('*/mhr-inputs.npz'))
    if not inputs:
        raise ValueError(f'No generated MHR inputs found in {directory}')
    faces = topology(model)
    for path in inputs:
        with np.load(path, allow_pickle=False) as saved:
            with torch.inference_mode():
                vertices, _ = model(
                    torch.from_numpy(saved['identity_coeffs']),
                    torch.from_numpy(saved['model_parameters']),
                    torch.from_numpy(saved['face_expr_coeffs']),
                )
            expected = vertices[0].double().numpy()
            expected[:, 1] -= float(saved['native_floor_y_cm'])
            expected *= float(saved['scale'])
        mesh_path = path.parent / 'body.ply'
        mesh = trimesh.load_mesh(mesh_path, process=False)
        error = float(np.abs(mesh.vertices - expected).max())
        report = json.loads((path.parent / 'fit-report.json').read_text(encoding='utf-8'))
        if not np.array_equal(faces, mesh.faces) or error > 5e-5:
            raise ValueError(f'Mesh does not reproduce: {path.parent.name}, {error} cm')
        if hashlib.sha256(mesh_path.read_bytes()).hexdigest() != report['mesh_sha256']:
            raise ValueError(f'Mesh checksum differs: {path.parent.name}')
        if abs(float(mesh.extents[1]) - report['height_target_cm']) > 5e-5:
            raise ValueError(f'Height differs: {path.parent.name}')
        reports.append({'subject_id': path.parent.name, 'maximum_vertex_error_cm': error, 'status': 'passed'})
    return reports
