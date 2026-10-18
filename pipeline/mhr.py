"""Inference using Meta's official MHR TorchScript release."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import trimesh


def load_model(path: Path, threads: int = 4):
    if not path.is_file():
        raise FileNotFoundError(f'{path}: run python scripts/download_mhr_assets.py first')
    torch.set_num_threads(threads)
    model = torch.jit.load(str(path), map_location='cpu').eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def topology(model) -> np.ndarray:
    return model.character_torch.mesh.faces.detach().cpu().numpy().astype(np.int64)


def evaluate(model, identity=None):
    identity = torch.zeros(1, 45) if identity is None else identity
    pose = torch.zeros(identity.shape[0], 204)
    expression = torch.zeros(identity.shape[0], 72)
    return model(identity, pose, expression)


def mesh_checks(vertices, faces) -> dict:
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Invalid mesh vertices')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid mesh topology')
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError('Invalid mesh surface area')
    return {
        'vertices': len(vertices), 'triangles': len(faces), 'finite': True,
        'extents_cm': mesh.extents.tolist(), 'watertight': bool(mesh.is_watertight),
        'surface_area_cm2': float(mesh.area),
    }


def smoke_test(model_path: Path, output: Path):
    model = load_model(model_path)
    with torch.inference_mode():
        vertices, skeleton = evaluate(model)
        changed = torch.zeros(1, 45)
        changed[0, 0] = 1
        varied, _ = evaluate(model, changed)
        repeated, _ = evaluate(model)
    faces = topology(model)
    vertices_np = vertices[0].numpy()
    report = mesh_checks(vertices_np, faces)
    if not torch.isfinite(skeleton).all():
        raise ValueError('Nonfinite skeleton')
    if not torch.equal(vertices, repeated):
        raise ValueError('Repeated inference differs')
    displacement = float((vertices - varied).abs().max())
    if displacement <= 1e-5:
        raise ValueError('Identity coefficients do not change the mesh')
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'neutral.ply'
    trimesh.Trimesh(vertices=vertices_np, faces=faces, process=False).export(path)
    reloaded = trimesh.load_mesh(path, process=False)
    if not np.allclose(reloaded.vertices, vertices_np, atol=1e-5) or not np.array_equal(reloaded.faces, faces):
        raise ValueError('Mesh export round trip differs')
    with model_path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    report.update({
        'status': 'passed', 'torch_version': torch.__version__, 'device': 'cpu',
        'model_sha256': digest, 'identity_input_shape': [1, 45],
        'pose_input_shape': [1, 204], 'expression_input_shape': [1, 72],
        'skeleton_shape': list(skeleton.shape), 'deterministic': True,
        'identity_response_max_cm': displacement, 'export_round_trip': 'passed',
        'coordinate_system': 'MHR native: Y up; centimetres',
    })
    (output / 'smoke-report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report
