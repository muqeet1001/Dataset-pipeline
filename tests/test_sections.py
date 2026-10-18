import numpy as np
import pytest
import torch
import trimesh
from pipeline.sections import section_perimeter, stature_transform


def cylinder(radius=2.0):
    mesh = trimesh.creation.cylinder(radius=radius, height=10, sections=64)
    mesh.vertices = mesh.vertices[:, [0, 2, 1]]
    return mesh


def test_section_matches_analytic_polygon_perimeter_and_gradient():
    mesh = cylinder()
    radius_scale = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    vertices = torch.from_numpy(np.asarray(mesh.vertices).copy())
    scale = torch.stack((radius_scale, radius_scale * 0 + 1, radius_scale))
    perimeter = section_perimeter(vertices * scale, mesh.faces, 0.123)
    expected = 2 * 64 * 2 * np.sin(np.pi / 64)
    assert float(perimeter.detach()) == pytest.approx(expected, abs=1e-8)
    perimeter.backward()
    assert float(radius_scale.grad) == pytest.approx(expected, abs=1e-8)


def test_disjoint_arm_sections_are_excluded():
    torso = cylinder()
    left = cylinder(1)
    right = cylinder(1)
    left.apply_translation([-10, 0, 0])
    right.apply_translation([10, 0, 0])
    mesh = trimesh.util.concatenate((torso, left, right))
    measured = section_perimeter(torch.from_numpy(np.asarray(mesh.vertices).copy()), mesh.faces, 0.123)
    assert float(measured) == pytest.approx(2 * 64 * 2 * np.sin(np.pi / 64), abs=1e-8)


def test_missing_section_fails():
    mesh = cylinder()
    with pytest.raises(ValueError, match='No mesh section'):
        section_perimeter(torch.from_numpy(np.asarray(mesh.vertices).copy()), mesh.faces, 99)


def test_stature_is_exact_and_grounded():
    mesh = cylinder()
    vertices, scale, floor = stature_transform(torch.from_numpy(np.asarray(mesh.vertices).copy()), 180)
    assert float(vertices[:, 1].min()) == pytest.approx(0)
    assert float(vertices[:, 1].max()) == pytest.approx(180)
    assert float(scale) == pytest.approx(18)
    assert float(floor) == pytest.approx(-5)
