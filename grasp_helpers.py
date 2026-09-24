"""
Utility functions used by multiple pipeline modules.

Contains:
  - PLY file reading (point cloud)
  - basic geometry operations (normalise, PCA)
  - farthest-point sampling
  - JSON I/O
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_open3d_binary_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a binary PLY file (Open3D format).

    A PLY file stores a 3D point cloud. Each point has:
      - x, y, z     = position in space
      - nx, ny, nz  = surface normal (perpendicular direction)
      - red, green, blue = colour

    Returns:
      points  (N, 3) — XYZ coordinates
      normals (N, 3) — surface normals
    """
    vertex_count = None
    with path.open("rb") as handle:
        # Read the text header until "end_header"
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Incomplete PLY header in {path}")
            if line.startswith(b"element vertex"):
                vertex_count = int(line.split()[2])
            if line.strip() == b"end_header":
                break

        if vertex_count is None:
            raise RuntimeError(f"Missing vertex count in {path}")

        # After the header comes binary data — read as a structured array
        dtype = np.dtype([
            ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),       # position
            ("nx", "<f8"), ("ny", "<f8"), ("nz", "<f8"),     # normal
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),  # colour
        ])
        data = np.fromfile(handle, dtype=dtype, count=vertex_count)

    points = np.column_stack((data["x"], data["y"], data["z"]))
    normals = np.column_stack((data["nx"], data["ny"], data["nz"]))
    return points, normals


def normalize(vector: np.ndarray) -> np.ndarray:
    """
    Turn a vector into a unit vector (length = 1).

    If the vector is near-zero, return it unchanged (avoid division by zero).
    """
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return vector.copy()
    return vector / norm


def project_sparse_point_heat_to_object(
    object_points: np.ndarray,
    patch_points: np.ndarray,
    patch_heat_values: np.ndarray,
) -> np.ndarray:
    """
    Spread sparse 3D heat values from patch points onto the full visible object.

    Each object point inherits the heat of its nearest patch point, with a
    gentle distance decay. This keeps the upstream hot region influential
    while still giving the whole observed surface a usable heat prior.
    """
    if len(object_points) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(patch_points) == 0 or len(patch_heat_values) != len(patch_points):
        return np.zeros((len(object_points),), dtype=np.float64)

    nearest_d2 = np.full(len(object_points), np.inf, dtype=np.float64)
    nearest_heat = np.zeros(len(object_points), dtype=np.float64)

    chunk = 256
    for start in range(0, len(object_points), chunk):
        end = min(start + chunk, len(object_points))
        block = object_points[start:end]
        diffs = block[:, None, :] - patch_points[None, :, :]
        d2 = np.einsum("bij,bij->bi", diffs, diffs)
        nn_idx = np.argmin(d2, axis=1)
        nearest_d2[start:end] = d2[np.arange(end - start), nn_idx]
        nearest_heat[start:end] = patch_heat_values[nn_idx]

    patch_extent = patch_points.max(axis=0) - patch_points.min(axis=0)
    sigma_m = max(float(np.linalg.norm(patch_extent)) * 0.35, 0.015)
    decay = np.exp(-nearest_d2 / max(sigma_m * sigma_m, 1e-9))
    return np.clip(nearest_heat * decay, 0.0, 1.0)


def farthest_point_sampling(points: np.ndarray, count: int) -> np.ndarray:
    """
    Pick 'count' points that are as far apart from each other as possible.

    Greedy algorithm:
      1. Start with the first point.
      2. At each step, add the point that is farthest from all
         already-selected points.

    This guarantees the seeds are spread uniformly across the surface,
    not clustered in one area.

    Returns: array of selected point indices.
    """
    if len(points) <= count:
        return np.arange(len(points))

    selected = [0]
    # Minimum distance from each point to any already-selected seed
    min_distances = np.linalg.norm(points - points[0], axis=1)

    for _ in range(count - 1):
        # The point whose minimum distance is largest = the most isolated
        next_idx = int(np.argmax(min_distances))
        selected.append(next_idx)
        # Update distances: each point keeps the distance to its
        # closest selected seed
        distances = np.linalg.norm(points - points[next_idx], axis=1)
        min_distances = np.minimum(min_distances, distances)

    return np.array(selected, dtype=int)


def pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Principal Component Analysis (PCA) on a 3D point set.

    PCA finds the 3 directions where the points vary the most:
      - eigenvector 0: direction of maximum variance (long axis)
      - eigenvector 1: second direction (medium axis)
      - eigenvector 2: direction of minimum variance (short axis = normal)

    For a flat surface, eigenvectors 0 and 1 define the plane,
    and eigenvector 2 is the surface normal.

    Returns:
      eigenvalues  (3,) — variance along each axis, descending
      eigenvectors (3,3) — columns are the principal directions
    """
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered.T)       # 3x3 covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # eigh returns ascending order, we want descending
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def save_json(path: Path, data: dict) -> None:
    """Save a dictionary as formatted JSON."""
    path.write_text(json.dumps(data, indent=2))
