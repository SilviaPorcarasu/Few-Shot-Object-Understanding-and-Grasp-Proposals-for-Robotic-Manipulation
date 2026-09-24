"""
Wrench-space grasp quality metrics.

Two variants available:

  opposition_score  — Friction-free, purely geometric.  Checks that the
                      contact normals "close in" from opposed sides.  No
                      friction assumptions needed.  Fast.

  wrench_score      — Friction-cone based.  For each contact, samples force
                      directions on the edges of the linearized friction cone
                      and assembles the 6×N grasp matrix G.  Tests whether a
                      set of canonical external perturbations (gravity, lateral
                      pushes, torques) can be resisted by non-negative contact
                      forces via NNLS.  Tested over multiple μ values so the
                      result is conservative when the object material is unknown
                      (Robotiq 3F has rubber fingertips → μ is determined by
                      the gripper, not the object).

Usage in score_grasps.py:
    from grasp_wrench import opposition_score, wrench_score
    opp  = opposition_score(pts, norms)
    wqs  = wrench_score(pts, norms, object_center, approach)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

# Default μ values to test (rubber fingertip on unknown object surface).
# Conservative lower bound μ=0.2 covers slippery smooth objects (glass,
# polished metal). μ=0.6 covers rough/textured surfaces. The wrench_score
# function returns the MINIMUM over all tested μ, so a grasp that only
# holds at high friction gets penalised.
DEFAULT_MU_VALUES: tuple[float, ...] = (0.2, 0.4, 0.6)

# Number of edges of the linearized friction cone per contact.
# 8 is the standard in analytical grasp quality literature; more edges
# give a tighter polytope but cost more NNLS solves.
FRICTION_CONE_EDGES: int = 8

# Residual tolerance for NNLS: if ||G λ - (-d)||/||d|| < this, the
# perturbation is considered "resisted".
_NNLS_RESIDUAL_TOL: float = 0.15


# ── Internal helpers ─────────────────────────────────────────────────

def _tangent_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to normal."""
    n = normal / (float(np.linalg.norm(normal)) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= float(np.linalg.norm(t1)) + 1e-12
    t2 = np.cross(n, t1)
    return t1, t2


def _sample_friction_cone(
    normal: np.ndarray,
    mu: float,
    num_edges: int = FRICTION_CONE_EDGES,
) -> np.ndarray:
    """
    Sample `num_edges` force directions on the boundary of the linearized
    friction cone.  Returns shape (num_edges, 3), each row a unit force
    vector inside the cone (f = n + μ*(cos θ·t1 + sin θ·t2)).
    """
    t1, t2 = _tangent_frame(normal)
    n = normal / (float(np.linalg.norm(normal)) + 1e-12)
    forces = np.empty((num_edges, 3), dtype=np.float64)
    for i in range(num_edges):
        angle = 2.0 * np.pi * i / num_edges
        f = n + mu * (np.cos(angle) * t1 + np.sin(angle) * t2)
        forces[i] = f / max(float(np.linalg.norm(f)), 1e-12)
    return forces


def _build_grasp_matrix(
    contact_points: list[np.ndarray],
    contact_normals: list[np.ndarray],
    object_center: np.ndarray,
    mu: float,
    num_edges: int = FRICTION_CONE_EDGES,
    char_length: float = 0.05,
) -> np.ndarray:
    """
    Assemble the 6×N grasp matrix where N = n_contacts * num_edges.

    Each column is a normalized wrench [f ; (r×f)/L]:
      - f       : sampled friction-cone force (unit vector)
      - r       : contact_point - object_center
      - L       : char_length (typical contact-to-center distance) normalizes
                  force and torque rows to the same numerical scale so that
                  NNLS treats them equally.
    """
    cols = []
    for p, n in zip(contact_points, contact_normals):
        r = p - object_center
        for f in _sample_friction_cone(n, mu, num_edges):
            torque = np.cross(r, f) / max(char_length, 1e-12)
            cols.append(np.concatenate([f, torque]))
    return np.array(cols, dtype=np.float64).T  # (6, N)


def _resists_perturbation(G: np.ndarray, perturbation: np.ndarray) -> bool:
    """
    Test if contact forces within the friction cone can resist `perturbation`.

    Finds λ ≥ 0 (NNLS) minimising ||G λ - (-perturbation)||.
    The grasp "resists" if the residual is small relative to the perturbation.
    """
    target = -perturbation
    target_norm = float(np.linalg.norm(target))
    if target_norm < 1e-12:
        return True
    _, residual = nnls(G, target)
    return (residual ** 0.5) / target_norm < _NNLS_RESIDUAL_TOL


def _build_test_perturbations(
    approach_direction: np.ndarray,
    char_length: float = 0.05,
) -> list[np.ndarray]:
    """
    Canonical set of external wrenches the grasp must resist.

    Forces:   gravity(-Z), left/right, forward/backward, along/against approach
    Torques:  roll (about approach), pitch, yaw  (both signs)

    Torque entries are divided by char_length to match the normalization in
    _build_grasp_matrix so force and torque rows are numerically comparable.
    """
    a = approach_direction / (float(np.linalg.norm(approach_direction)) + 1e-12)

    # Build a right-handed lateral frame
    ref = np.array([0.0, 0.0, 1.0])
    lat = np.cross(a, ref)
    lat_norm = float(np.linalg.norm(lat))
    if lat_norm < 0.1:
        ref = np.array([1.0, 0.0, 0.0])
        lat = np.cross(a, ref)
        lat_norm = float(np.linalg.norm(lat))
    lat /= max(lat_norm, 1e-12)
    fwd = np.cross(lat, a)
    fwd /= float(np.linalg.norm(fwd)) + 1e-12

    gravity = np.array([0.0, 0.0, -1.0])

    zero3 = np.zeros(3)
    perturbations: list[np.ndarray] = []

    # ── Force perturbations ──
    for d in [gravity, lat, -lat, fwd, -fwd, a, -a]:
        perturbations.append(np.concatenate([d, zero3]))

    # ── Torque perturbations (normalized) ──
    for axis in [a, lat, fwd]:
        perturbations.append(np.concatenate([zero3, axis]))
        perturbations.append(np.concatenate([zero3, -axis]))

    return perturbations


# ── Public API ───────────────────────────────────────────────────────

def opposition_score(
    contact_points: list[np.ndarray],
    contact_normals: list[np.ndarray],
) -> float:
    """
    Friction-free geometric opposition quality for the Robotiq 3F
    1-thumb + 2-fingers opposed configuration.

    Measures how well contact normals "close in" on the object:
      - thumb normal  → should point toward the two-finger midpoint
      - finger normals → should point toward the thumb

    Returns a value in [0, 1].  No friction coefficient assumed.
    """
    if len(contact_points) < 3 or len(contact_normals) < 3:
        return 0.0

    p1, p2, p3 = contact_points[:3]
    n1, n2, n3 = contact_normals[:3]

    n1u = n1 / (float(np.linalg.norm(n1)) + 1e-12)
    n2u = n2 / (float(np.linalg.norm(n2)) + 1e-12)
    n3u = n3 / (float(np.linalg.norm(n3)) + 1e-12)

    fingers_mid = 0.5 * (p2 + p3)
    diff = fingers_mid - p1
    dist = float(np.linalg.norm(diff))
    if dist < 1e-6:
        return 0.0
    thumb_to_fingers = diff / dist
    fingers_to_thumb = -thumb_to_fingers

    s_thumb = max(0.0, float(n1u @ thumb_to_fingers))
    s_f2    = max(0.0, float(n2u @ fingers_to_thumb))
    s_f3    = max(0.0, float(n3u @ fingers_to_thumb))

    return float((s_thumb + s_f2 + s_f3) / 3.0)


def wrench_score(
    contact_points: list[np.ndarray],
    contact_normals: list[np.ndarray],
    object_center: np.ndarray,
    approach_direction: np.ndarray,
    mu_values: tuple[float, ...] = DEFAULT_MU_VALUES,
    num_edges: int = FRICTION_CONE_EDGES,
) -> float:
    """
    Wrench-space grasp quality score.

    For each μ in mu_values:
      1. Sample the linearized friction cone for every contact.
      2. Build grasp matrix G (6 × N).
      3. For each canonical perturbation, test via NNLS whether
         non-negative contact forces can resist it.
      4. score_μ = n_resisted / n_total_perturbations

    Returns min(scores over all μ): conservative for unknown materials.
    Value in [0, 1]; 1 = all perturbations resisted even at μ = 0.2.
    """
    if len(contact_points) < 3 or len(contact_normals) < 3:
        return 0.0

    # Characteristic length = mean contact-to-center distance
    char_length = float(np.mean([
        float(np.linalg.norm(p - object_center)) for p in contact_points
    ]))
    char_length = max(char_length, 0.01)

    test_perturbations = _build_test_perturbations(approach_direction, char_length)
    n_tests = len(test_perturbations)

    scores_per_mu: list[float] = []
    for mu in mu_values:
        G = _build_grasp_matrix(
            contact_points, contact_normals, object_center,
            mu, num_edges, char_length,
        )
        n_resisted = sum(
            _resists_perturbation(G, p) for p in test_perturbations
        )
        scores_per_mu.append(n_resisted / n_tests)

    return float(min(scores_per_mu))
