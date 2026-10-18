"""Fit MHR body identity to three measured torso-section proxies and stature."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import torch
import trimesh
from .mhr import evaluate, mesh_checks, topology
from .sections import section_perimeter, stature_transform

TARGETS = (
    ('chestcircumference', 'chestheight'),
    ('waistcircumference', 'waistheightomphalion'),
    ('buttockcircumference', 'buttockheight'),
)


class BodyFitter:
    def __init__(self, model, max_error_cm=2.0):
        self.model = model
        self.faces = topology(model)
        self.max_error_cm = max_error_cm
        # At fixed pose/expression the official model is affine in identity.
        # Build an exact local basis, then verify against official inference at export.
        identity = torch.zeros(21, 45)
        identity[1:, :20] = torch.eye(20)
        with torch.inference_mode():
            uncorrected, _ = model(identity, torch.zeros(21, 204), torch.zeros(21, 72), False)
            base, _ = evaluate(model)
        self.base = base[0].double().detach().clone()
        self.basis = (uncorrected[1:] - uncorrected[0]).double().detach().clone()

    def fit(self, subject: dict, output: Path, max_evaluations=150):
        started = time.perf_counter()
        measurements = subject['measurements_cm']
        target_height = measurements['stature']
        targets = np.array([measurements[name] for name, _ in TARGETS])
        heights = [measurements[height] for _, height in TARGETS]
        if not all(0 < height < target_height for height in heights):
            raise ValueError('Section heights must lie within stature')

        def residual_and_jacobian(coefficients):
            coefficients_t = torch.tensor(coefficients, dtype=torch.float64, requires_grad=True)
            vertices = self.base + torch.einsum('i,ijk->jk', coefficients_t, self.basis)
            vertices, _, _ = stature_transform(vertices, target_height)
            perimeters = torch.stack([section_perimeter(vertices, self.faces, height) for height in heights])
            errors = perimeters - torch.tensor(targets)
            jacobian = torch.stack([
                torch.autograd.grad(error, coefficients_t, retain_graph=True)[0]
                for error in errors
            ]).detach().numpy()
            # Minimum-norm identity prior reduces underdetermination. Body PCA is bounded.
            prior_weight = 0.05
            residual = np.concatenate((errors.detach().numpy(), prior_weight * coefficients))
            jacobian = np.concatenate((jacobian, prior_weight * np.eye(20)), axis=0)
            return residual, jacobian

        last = None

        def cached(values):
            nonlocal last
            if last is None or not np.array_equal(last[0], values):
                last = (values.copy(), residual_and_jacobian(values))
            return last[1]

        initial, _ = cached(np.zeros(20))
        solution = least_squares(
            lambda values: cached(values)[0], np.zeros(20),
            jac=lambda values: cached(values)[1], bounds=(-3.0, 3.0),
            max_nfev=max_evaluations, ftol=1e-8, xtol=1e-8, gtol=1e-7,
        )
        identity = torch.zeros(1, 45)
        identity[0, :20] = torch.from_numpy(solution.x.astype(np.float32))
        with torch.inference_mode():
            native, skeleton = evaluate(self.model, identity)
        predicted = self.base + torch.einsum('i,ijk->jk', torch.from_numpy(solution.x), self.basis)
        official_error = float((native[0].double() - predicted).abs().max())
        if official_error > 1e-3:
            raise ValueError(f'Fitting basis disagrees with official MHR: {official_error} cm')
        vertices, scale, floor = stature_transform(native[0].double(), target_height)
        achieved = np.array([float(section_perimeter(vertices, self.faces, height)) for height in heights])
        errors = achieved - targets
        checks = mesh_checks(vertices.numpy(), self.faces)
        accepted = bool(solution.success and np.max(np.abs(errors)) <= self.max_error_cm and checks['watertight'])
        output.mkdir(parents=True, exist_ok=True)
        mesh = trimesh.Trimesh(vertices=vertices.numpy(), faces=self.faces, process=False)
        mesh_path = output / 'body.ply'
        mesh.export(mesh_path)
        # Store exact official inputs, the external similarity transform and the skeleton.
        np.savez_compressed(
            output / 'mhr-inputs.npz', identity_coeffs=identity.numpy(),
            model_parameters=np.zeros((1, 204), dtype=np.float32),
            face_expr_coeffs=np.zeros((1, 72), dtype=np.float32),
            scale=np.array(float(scale)), native_floor_y_cm=np.array(float(floor)),
            skeleton_state_native=skeleton.numpy(),
        )
        (output / 'source-record.json').write_text(json.dumps(subject, indent=2) + '\n', encoding='utf-8')
        report = {
            'subject_id': subject['subject_id'],
            'status': 'accepted_proxy_fit' if accepted else 'rejected_fit',
            'training_label_kind': 'estimated_MHR_body_from_anthropometry',
            'source_commit': subject['source_commit'],
            'model_sha256': '352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc',
            'model_backend': 'official MHR v1.0.1 TorchScript; CPU; LOD 1',
            'mesh': checks,
            'height_target_cm': target_height,
            'height_achieved_cm': float(vertices[:, 1].max() - vertices[:, 1].min()),
            'uniform_scale': float(scale),
            'measurement_method': 'closed central horizontal mesh sections at ANSUR chest, omphalion-waist and buttock heights; anatomical correspondence is approximate',
            'measurements': {
                name: {
                    'target_cm': float(targets[index]), 'achieved_cm': float(achieved[index]),
                    'error_cm': float(errors[index]), 'section_height_cm': float(heights[index]),
                    'before_fit_error_cm': float(initial[index]),
                }
                for index, (name, _) in enumerate(TARGETS)
            },
            'max_allowed_proxy_error_cm': self.max_error_cm,
            'optimized_inputs': 'first 20 MHR body identity coefficients, bounded [-3,3]',
            'unfitted': [name for name in measurements if name not in {'stature', *(name for pair in TARGETS for name in pair)}],
            'mass_kg_target': subject['mass_kg'], 'mass_fitted': False,
            'pose': 'MHR neutral pose; zero 204 model parameters; zero facial expression',
            'official_forward_max_difference_cm': official_error,
            'solver_success': bool(solution.success), 'solver_message': str(solution.message),
            'evaluations': solution.nfev, 'elapsed_seconds': time.perf_counter() - started,
            'mesh_sha256': hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
            'limitations': [
                'Selected measurements underdetermine a full 3D body; this is an estimate.',
                'Cross-section proxies have not been validated against ANSUR anatomical landmarks.',
                'Remaining measurements and body mass are preserved as targets but are not fitted.',
            ],
        }
        (output / 'fit-report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        return report
