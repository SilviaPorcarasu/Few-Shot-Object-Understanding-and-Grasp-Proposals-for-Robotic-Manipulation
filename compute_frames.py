"""
STAGE 2: Build a local coordinate system on each patch.

The normal is flipped to point TOWARD the camera. Why? Because the gripper
camera is mounted on the gripper, so the surface visible to the camera is
the surface we can grasp.

Input:  list of Patch + point cloud + camera origin
Output: list of LocalFrame
"""

from __future__ import annotations

import numpy as np

from grasp_helpers import normalize, pca_basis
from grasp_types import LocalFrame, Patch


def compute_frame(
    patch: Patch,
    points: np.ndarray,
    camera_origin: np.ndarray,
) -> LocalFrame:
    """
    Build the PCA frame for a single patch.

    1. Compute the centroid (mean) of the patch points.
    2. PCA → eigenvectors (principal directions).
    3. The eigenvector with the smallest variance = surface normal.
    4. Flip the normal so it points toward the camera (the surface
       "looks back" at the camera, since that's the side we can grasp).
    5. Rebuild tangent_2 via cross product (guarantees orthonormality).
    """
    patch_points = points[patch.point_indices]
    centroid = patch_points.mean(axis=0)
    eigenvalues, eigenvectors = pca_basis(patch_points)

    tangent_1 = normalize(eigenvectors[:, 0])  # principal direction 1
    normal = normalize(eigenvectors[:, 2])      # smallest-variance direction

    # Vector from patch toward the camera. Normal must agree with this.
    to_camera = normalize(camera_origin - centroid)
    if float(normal @ to_camera) < 0.0:
        normal = -normal

    # Rebuild tangent_2 to be perpendicular to both normal and tangent_1.
    tangent_2 = normalize(np.cross(normal, tangent_1))
    tangent_1 = normalize(np.cross(tangent_2, normal))

    # Local elongation in the patch plane: 1 - lambda2/lambda1.
    # High values flag locally elongated regions (cup handle, bottle
    # neck, ridge of an object); low values flag isotropic patches
    # (flat body of a cup, side of a sphere or cube). Used both for
    # scoring (feature-rich regions are usually a better grip) and to
    # decide whether to take the long axis from the local patch or
    # from the global object PCA (see generate_contacts).
    lam1 = float(max(eigenvalues[0], 1e-12))
    lam2 = float(max(eigenvalues[1], 0.0))
    local_elongation = float(np.clip(1.0 - (lam2 / lam1), 0.0, 1.0))

    return LocalFrame(
        patch=patch,
        centroid=centroid,
        tangent_1=tangent_1,
        tangent_2=tangent_2,
        normal=normal,
        pca_eigenvalues=eigenvalues,
        local_elongation=local_elongation,
    )


def compute_frames(
    patches: list[Patch],
    points: np.ndarray,
    camera_origin: np.ndarray,
) -> list[LocalFrame]:
    """Compute a local PCA frame for every patch."""
    return [compute_frame(p, points, camera_origin) for p in patches]
