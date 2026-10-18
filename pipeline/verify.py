"""Reproduce stored meshes from their exact official model inputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import trimesh
from .mhr import topology


def verify_subject(model, directory: Path):
    faces = topology(model)
    with np.load(directory / 'mhr-inputs.npz', allow_pickle=False) as saved:
        with torch.inference_mode():
            vertices, _ = model(
                torch.from_numpy(saved['identity_coeffs']),
                torch.from_numpy(saved['model_parameters']),
                torch.from_numpy(saved['face_expr_coeffs']),
            )
        expected = vertices[0].double().numpy()
        expected[:, 1] -= float(saved['native_floor_y_cm'])
        expected *= float(saved['scale'])
    mesh_path = directory / 'body.ply'
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if mesh.vertices.shape != expected.shape or not np.isfinite(mesh.vertices).all():
        raise ValueError(f'Invalid mesh vertices: {directory.name}')
    error = float(np.abs(mesh.vertices - expected).max())
    report = json.loads((directory / 'fit-report.json').read_text(encoding='utf-8'))
    if not np.array_equal(faces, mesh.faces) or not np.isfinite(error) or error > 5e-5:
        raise ValueError(f'Mesh does not reproduce: {directory.name}, {error} cm')
    if hashlib.sha256(mesh_path.read_bytes()).hexdigest() != report['mesh_sha256']:
        raise ValueError(f'Mesh checksum differs: {directory.name}')
    if abs(float(mesh.extents[1]) - report['height_target_cm']) > 5e-5:
        raise ValueError(f'Height differs: {directory.name}')
    return {'subject_id': directory.name, 'maximum_vertex_error_cm': error, 'status': 'passed'}


def verify_generated(model, directory: Path):
    inputs = sorted(directory.glob('*/mhr-inputs.npz'))
    if not inputs:
        raise ValueError(f'No generated MHR inputs found in {directory}')
    return [verify_subject(model, path.parent) for path in inputs]
