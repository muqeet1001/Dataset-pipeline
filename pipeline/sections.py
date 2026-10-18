"""Closed central mesh cross sections; perimeter remains differentiable."""
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def central_section_edges(vertices: np.ndarray, faces: np.ndarray, height: float):
    """Return the two crossed mesh edges for each segment of the torso loop.

    Components are separated using mesh edge IDs. Arms and other disjoint loops
    cannot add to the torso circumference. A loop must be closed and span the
    body's sagittal plane, otherwise the measurement is rejected.
    """
    triangle_y = vertices[faces, 1]
    above = triangle_y > height
    crosses = above != np.roll(above, -1, axis=1)
    triangles = faces[crosses.sum(axis=1) == 2]
    crosses = crosses[crosses.sum(axis=1) == 2]
    if not len(triangles):
        raise ValueError(f'No mesh section at {height:.3f} cm')
    edges = np.stack((triangles, np.roll(triangles, -1, axis=1)), axis=-1)[crosses].reshape(-1, 2, 2)
    sorted_edges = np.sort(edges, axis=-1)
    codes = sorted_edges[:, :, 0] * len(vertices) + sorted_edges[:, :, 1]
    _, inverse = np.unique(codes, return_inverse=True)
    nodes = inverse.reshape(-1, 2)
    adjacency = coo_matrix(
        (np.ones(nodes.size), (np.concatenate((nodes[:, 0], nodes[:, 1])), np.concatenate((nodes[:, 1], nodes[:, 0])))),
        shape=(inverse.max() + 1, inverse.max() + 1),
    )
    _, labels = connected_components(adjacency, directed=False)
    segment_labels = labels[nodes[:, 0]]
    a, b = vertices[edges[:, :, 0]], vertices[edges[:, :, 1]]
    fractions = (height - a[:, :, 1]) / (b[:, :, 1] - a[:, :, 1])
    points = a + fractions[:, :, None] * (b - a)
    mid_x = (vertices[:, 0].min() + vertices[:, 0].max()) / 2
    candidates = []
    for label in np.unique(segment_labels):
        indices = np.flatnonzero(segment_labels == label)
        ring = points[indices]
        if not (ring[:, :, 0].min() < mid_x < ring[:, :, 0].max()):
            continue
        _, degree = np.unique(nodes[indices], return_counts=True)
        if not np.all(degree == 2):
            continue
        perimeter = np.linalg.norm(ring[:, 1] - ring[:, 0], axis=1).sum()
        candidates.append((float(perimeter), indices))
    if not candidates:
        raise ValueError(f'No closed central torso section at {height:.3f} cm')
    # A second central closed surface (e.g. clothing) is ambiguous for body fitting.
    if len(candidates) != 1:
        raise ValueError(f'Ambiguous central torso section at {height:.3f} cm')
    return edges[candidates[0][1]]


def section_perimeter(vertices: torch.Tensor, faces: np.ndarray, height: float):
    edges = central_section_edges(vertices.detach().cpu().numpy(), faces, height)
    a = vertices[edges[:, :, 0]]
    b = vertices[edges[:, :, 1]]
    fraction = (height - a[:, :, 1]) / (b[:, :, 1] - a[:, :, 1])
    points = a + fraction[:, :, None] * (b - a)
    return torch.linalg.vector_norm(points[:, 1] - points[:, 0], dim=1).sum()


def stature_transform(vertices: torch.Tensor, height_cm: float):
    lowest = vertices[:, 1].min()
    extent = vertices[:, 1].max() - lowest
    scale = height_cm / extent
    offset = torch.stack((lowest * 0, lowest, lowest * 0))
    return (vertices - offset) * scale, scale, lowest
