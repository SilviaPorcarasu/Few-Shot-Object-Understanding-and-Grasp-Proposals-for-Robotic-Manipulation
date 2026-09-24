"""
STAGE 3: Place 3 finger contacts using the Robotiq 3F's "1-thumb + 2-fingers
opposed" configuration.

The Robotiq 3F has finger A (thumb) opposed to fingers B and C. We mirror
that here: one contact on one side of the object's long axis, two on the
opposite side, separated along the length. This spreads the contacts over
the object instead of clustering them on a single cross-section — which is
how a real hand or a real Robotiq 3F actually grasps an elongated object
(banana, bottle, telecomandă, ...).

For near-isotropic objects (sphere, cube) the projection of the principal
axis onto the patch plane is small, so the placement degenerates back to
"thumb on one side, fingers on the other" with very small longitudinal
spread — still a valid opposed grip.

Input:  frames + full object cloud (points / pixels / normals) + heatmap
        + camera origin + object's principal axis
Output: list of GraspCandidate
"""

from __future__ import annotations

import numpy as np

from grasp_config import (
    CONTACT_BOTTOM_EXCLUSION_FRACTION,
    CONTACT_UP_AXIS_CAMERA,
    CONTACT_LONG_TOLERANCE_M,
    CONTACT_PLANE_HALF_THICKNESS_M,
    CONTACT_RIM_FRACTION,
    CONTACT_RIM_FRACTION_FLEXIBLE,
    FLEXIBLE_DIAMETER_COMPRESSION,
    FLEXIBLE_LONG_TOLERANCE_SCALE,
    FLEXIBLE_SPREAD_SCALE,
    FINGER_SPREAD_CANDIDATES_M,
    FINGER_SPREAD_LONG_M,
    GRIPPER_MODE_BASIC,
    GRIPPER_MODE_PARAMS,
    GRIPPER_MODES,
    HEATMAP_CONTACT_BIAS,
    LOCAL_ELONGATION_THRESHOLD,
    ROTATION_CANDIDATES_DEG,
    USE_FLEXIBILITY,
)
from grasp_helpers import normalize
from grasp_types import Contact, GraspCandidate, LocalFrame
from select_patches import _visible_edge_scores_from_pixels


def _heatmap_at(heatmap: np.ndarray | None, pixel: tuple[int, int]) -> float:
    """Look up the heatmap value at a pixel (returns 0..1, or 0 if missing)."""
    if heatmap is None:
        return 0.0
    u, v = pixel
    h, w = heatmap.shape[:2]
    if 0 <= v < h and 0 <= u < w:
        return float(heatmap[v, u]) / 255.0
    return 0.0


def _contact_heat_value(
    heatmap: np.ndarray | None,
    point_heat_values: np.ndarray | None,
    point_index: int,
    pixel: tuple[int, int],
) -> float:
    """
    Return the ML affordance value for one chosen contact.

    Two sources are supported:
      1. A classic 2D heatmap image + pixel lookup
      2. A direct per-point 3D heat value already attached to the cloud

    The second path is what we use for fused asymmetric reconstructions:
    there is no single reliable 2D image to sample from anymore, but the
    upstream pipeline already gives us a heat value for every 3D patch point.
    """
    if point_heat_values is not None and 0 <= point_index < len(point_heat_values):
        return float(point_heat_values[point_index])
    return _heatmap_at(heatmap, pixel)


def _patch_local_axes(
    frame: LocalFrame, object_principal_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build two in-plane axes for the patch:
      tangent_long  — "along the object/feature" in the patch plane
      tangent_perp  — perpendicular to it (gripper closing direction).

    The choice of tangent_long is adaptive:
      - Locally elongated patches (handle of a cup, neck of a bottle,
        a ridge): use the PATCH's own PCA tangent_1 — it captures the
        local feature direction, which the gripper should align with.
      - Locally isotropic patches (flat body of a cup, side of a sphere
        or cube): fall back to the projection of the OBJECT's principal
        axis. For elongated objects this still gives a meaningful
        across-the-axis grip; for fully isotropic shapes the projection
        is small and we land on tangent_1 anyway.
    """
    normal = frame.normal

    if frame.local_elongation >= LOCAL_ELONGATION_THRESHOLD:
        tangent_long = frame.tangent_1
    else:
        proj = (
            object_principal_axis
            - float(object_principal_axis @ normal) * normal
        )
        if float(np.linalg.norm(proj)) < 1e-4:
            tangent_long = frame.tangent_1
        else:
            tangent_long = normalize(proj)

    tangent_perp = normalize(np.cross(normal, tangent_long))
    # Re-orthogonalise tangent_long to ensure exact orthonormality.
    tangent_long = normalize(np.cross(tangent_perp, normal))
    return tangent_long, tangent_perp


def _rotate_around_axis(
    vector: np.ndarray,
    axis: np.ndarray,
    theta_rad: float,
) -> np.ndarray:
    """
    Rotate one vector around a unit axis using Rodrigues' formula.

    We use this to rotate the gripper frame around the approach axis while
    keeping the same patch and the same approach side on the object.
    """
    axis = normalize(axis)
    return normalize(
        vector * np.cos(theta_rad)
        + np.cross(axis, vector) * np.sin(theta_rad)
        + axis * float(axis @ vector) * (1.0 - np.cos(theta_rad))
    )


def _find_extreme_point(
    coord_long: np.ndarray,
    coord_perp: np.ndarray,
    valid_idx: np.ndarray,
    target_long: float,
    long_tolerance: float,
    sign_perp: int,
    point_edge_scores: np.ndarray | None = None,
    point_heat_values: np.ndarray | None = None,
    rim_fraction: float = CONTACT_RIM_FRACTION,
) -> int | None:
    """
    Among `valid_idx` cloud points, pick a contact on the rim/silhouette.

    Steps:
      1. Restrict to points within ±long_tolerance of the target along
         the longitudinal axis.
      2. Keep only the points whose perpendicular coordinate falls in
         the outermost `rim_fraction` of values on the requested side
         (top 10% for sign_perp=+1, bottom 10% for sign_perp=-1) — this
         is the visible silhouette, not the smooth middle.
      3. Among those rim points, pick the one closest to target_long
         so the contact lands at the desired longitudinal offset.

    Using a percentile band instead of a single argmax makes the choice
    robust against noisy cloud points near the rim, and avoids picking
    a contact on the smooth interior of the visible surface.
    """
    long_window = np.abs(coord_long[valid_idx] - target_long) < long_tolerance
    cand = valid_idx[long_window]
    if cand.size == 0:
        return None

    perp_vals = coord_perp[cand]
    rim_pct = float(np.clip(rim_fraction, 0.01, 0.5))
    if sign_perp > 0:
        threshold = float(np.quantile(perp_vals, 1.0 - rim_pct))
        rim_mask = perp_vals >= threshold
    else:
        threshold = float(np.quantile(perp_vals, rim_pct))
        rim_mask = perp_vals <= threshold

    rim_cand = cand[rim_mask]
    if rim_cand.size == 0:
        rim_cand = cand

    long_diffs = np.abs(coord_long[rim_cand] - target_long)
    proximity_scores = 1.0 - np.clip(
        long_diffs / max(long_tolerance, 1e-9), 0.0, 1.0,
    )
    heat_scores = None
    if point_heat_values is not None and len(point_heat_values) == len(coord_long):
        heat_scores = np.clip(point_heat_values[rim_cand], 0.0, 1.0)

    if point_edge_scores is None or len(point_edge_scores) != len(coord_long):
        if heat_scores is None:
            return int(rim_cand[int(np.argmin(long_diffs))])
        combined = (1.0 - HEATMAP_CONTACT_BIAS) * proximity_scores + HEATMAP_CONTACT_BIAS * heat_scores
        return int(rim_cand[int(np.argmax(combined))])

    edge_scores = point_edge_scores[rim_cand]
    if heat_scores is None:
        combined = 0.65 * edge_scores + 0.35 * proximity_scores
    else:
        heat_w = float(np.clip(HEATMAP_CONTACT_BIAS, 0.0, 0.5))
        edge_w = 0.65 - 0.5 * heat_w
        prox_w = 1.0 - edge_w - heat_w
        combined = edge_w * edge_scores + prox_w * proximity_scores + heat_w * heat_scores
    return int(rim_cand[int(np.argmax(combined))])


def generate_contacts_for_frame(
    frame: LocalFrame,
    points: np.ndarray,
    pixels: np.ndarray,
    normals: np.ndarray,
    camera_origin: np.ndarray,
    heatmap: np.ndarray | None,
    object_principal_axis: np.ndarray,
    point_heat_values: np.ndarray | None = None,
    point_edge_scores: np.ndarray | None = None,
    finger_spread_m: float = FINGER_SPREAD_LONG_M,
    gripper_mode: str = GRIPPER_MODE_BASIC,
    rotation_deg: float = 0.0,
    object_flexible: bool = False,
) -> GraspCandidate | None:
    """
    Build a 3-contact grasp candidate using 1-thumb + 2-fingers opposed.

    Steps:
      1. Build (tangent_long, tangent_perp) on the patch.
      2. Express the WHOLE object cloud in (long, perp, normal) coords
         relative to the patch centroid.
      3. Keep only points that are (a) within a thin slab around the
         patch tangent plane, and (b) facing the camera.
      4. Pick:
           thumb (F1)   — minimum perp at along=0     (one side of object)
           finger (F2)  — maximum perp at along=+ΔL/2 (opposite side, +half spread)
           finger (F3)  — maximum perp at along=−ΔL/2 (opposite side, −half spread)

    The contacts are spread along the object's long axis, NOT clustered
    on the same cross-section.
    """
    centroid = frame.centroid
    # We only trust the OBSERVED surface. So the default approach stays tied
    # to the current view direction instead of assuming a fully reliable 3D
    # reconstruction around the whole object.
    camera_approach_axis = normalize(centroid - camera_origin)

    tangent_long_0, tangent_perp_0 = _patch_local_axes(
        frame, object_principal_axis,
    )
    theta_rad = np.deg2rad(rotation_deg)
    tangent_long = _rotate_around_axis(
        tangent_long_0, camera_approach_axis, theta_rad,
    )
    tangent_perp = _rotate_around_axis(
        tangent_perp_0, camera_approach_axis, theta_rad,
    )
    normal = frame.normal

    offsets = points - centroid
    coord_long = offsets @ tangent_long
    coord_perp = offsets @ tangent_perp
    coord_norm = offsets @ normal

    # Slab around the patch tangent plane (depth direction).
    near_plane = np.abs(coord_norm) < CONTACT_PLANE_HALF_THICKNESS_M

    # Camera-facing check: each cloud point's normal must point toward the
    # camera (otherwise it is a back-facing or noisy estimate).
    cam_vec = camera_origin - points
    cam_norm = np.linalg.norm(cam_vec, axis=1, keepdims=True)
    cam_norm[cam_norm < 1e-9] = 1.0
    cam_dir = cam_vec / cam_norm
    visible = np.einsum("ij,ij->i", normals, cam_dir) > 0.0

    valid_mask = near_plane & visible
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size < 3:
        return None

    # Exclude the bottom fraction of the object to avoid grasping from below.
    # "Up" is fixed in the camera frame so the constraint stays stable and
    # does not rotate with the current view direction.
    if CONTACT_BOTTOM_EXCLUSION_FRACTION > 0.0:
        up = normalize(np.asarray(CONTACT_UP_AXIS_CAMERA, dtype=np.float64))
        heights = points @ up
        valid_heights = heights[valid_idx]
        h_min = float(np.min(valid_heights))
        h_max = float(np.max(valid_heights))
        h_threshold = h_min + CONTACT_BOTTOM_EXCLUSION_FRACTION * (h_max - h_min)
        valid_idx = valid_idx[valid_heights >= h_threshold]
        if valid_idx.size < 3:
            return None

    if point_edge_scores is None and len(pixels) == len(points) and len(pixels) > 0:
        point_edge_scores = _visible_edge_scores_from_pixels(pixels)

    # Per-mode geometric placement parameters (basic / wide / pinch /
    # scissor). The mode reshapes how the 3 contacts are spread.
    mode_params = GRIPPER_MODE_PARAMS.get(
        gripper_mode, GRIPPER_MODE_PARAMS[GRIPPER_MODE_BASIC],
    )
    flex_active = bool(object_flexible and USE_FLEXIBILITY)
    effective_spread = finger_spread_m * mode_params["long_spread_scale"]
    spread_half = effective_spread / 2.0
    long_tolerance = CONTACT_LONG_TOLERANCE_M * (
        FLEXIBLE_LONG_TOLERANCE_SCALE if flex_active else 1.0
    )

    # Flexible objects deform around the fingers, so contacts don't need to
    # land exactly on the outermost rim — a wider percentile band is used.
    rim_fraction = (
        CONTACT_RIM_FRACTION_FLEXIBLE if flex_active else CONTACT_RIM_FRACTION
    )

    f1_idx = _find_extreme_point(
        coord_long, coord_perp, valid_idx,
        target_long=0.0, long_tolerance=long_tolerance,
        sign_perp=-1,
        point_edge_scores=point_edge_scores,
        point_heat_values=point_heat_values,
        rim_fraction=rim_fraction,
    )
    f2_idx = _find_extreme_point(
        coord_long, coord_perp, valid_idx,
        target_long=+spread_half, long_tolerance=long_tolerance,
        sign_perp=+1,
        point_edge_scores=point_edge_scores,
        point_heat_values=point_heat_values,
        rim_fraction=rim_fraction,
    )
    f3_idx = _find_extreme_point(
        coord_long, coord_perp, valid_idx,
        target_long=-spread_half, long_tolerance=long_tolerance,
        sign_perp=+1,
        point_edge_scores=point_edge_scores,
        point_heat_values=point_heat_values,
        rim_fraction=rim_fraction,
    )
    if f1_idx is None or f2_idx is None or f3_idx is None:
        return None

    contacts: list[Contact] = []
    chosen_indices = [f1_idx, f2_idx, f3_idx]
    along_targets = [0.0, +spread_half, -spread_half]
    for i, (gi, target_long) in enumerate(
        zip(chosen_indices, along_targets), start=1,
    ):
        contact_pt = points[gi]
        contact_n = normalize(normals[gi])
        # On partial view-dependent reconstructions, it is safer to orient
        # contact normals toward the visible side seen by the camera.
        cam_to_contact = normalize(camera_origin - contact_pt)
        if float(contact_n @ cam_to_contact) < 0.0:
            contact_n = -contact_n
        pixel = (int(pixels[gi, 0]), int(pixels[gi, 1]))
        hm_val = _contact_heat_value(
            heatmap, point_heat_values, gi, pixel,
        )

        # Distance to the ideal placement target — useful as a sanity
        # value for downstream debugging (kept under the existing
        # `distance_to_visible_m` field).
        ideal_target = (
            centroid
            + target_long * tangent_long
            + (
                # signed half-diameter along tangent_perp; we don't know
                # the half-diameter a priori, so just store the offset to
                # the centroid of the cluster of valid points instead.
                0.0
            ) * tangent_perp
        )
        dist = float(np.linalg.norm(contact_pt - ideal_target))

        contacts.append(Contact(
            name=f"contact_{i}",
            angle_deg=float(target_long * 1000.0),  # store along-axis offset (mm) as a diag
            point=contact_pt,
            normal=contact_n,
            visible=True,
            distance_to_visible_m=dist,
            pixel=pixel,
            heatmap_value=hm_val,
        ))

    grasp_center = np.mean([c.point for c in contacts], axis=0)
    approach = normalize(grasp_center - camera_origin)

    return GraspCandidate(
        frame=frame,
        approach_direction=approach,
        contacts=contacts,
        gripper_mode=gripper_mode,
        finger_spread_m=finger_spread_m,
        rotation_deg=float(rotation_deg),
    )


def generate_contacts(
    frames: list[LocalFrame],
    points: np.ndarray,
    pixels: np.ndarray,
    normals: np.ndarray,
    camera_origin: np.ndarray,
    heatmap: np.ndarray | None,
    object_principal_axis: np.ndarray,
    point_heat_values: np.ndarray | None = None,
    object_flexible: bool = False,
) -> list[GraspCandidate]:
    """
    Generate contact triplets for every local frame.

    For every frame we try several finger spreads (defined by
    FINGER_SPREAD_CANDIDATES_M). Each spread becomes a separate
    candidate. The scoring stage decides which one wins, so the gripper
    aperture auto-adapts to the size of the local feature: a small
    spread fits a cup handle or a thin neck, a wider spread fits the
    body of a bottle or a banana.
    """
    candidates = []
    cached_edge_scores = None
    if len(pixels) == len(points) and len(pixels) > 0:
        cached_edge_scores = _visible_edge_scores_from_pixels(pixels)
    flex_active = bool(object_flexible and USE_FLEXIBILITY)
    spread_candidates = list(FINGER_SPREAD_CANDIDATES_M)
    if flex_active:
        expanded_spreads = [
            round(spread * FLEXIBLE_SPREAD_SCALE, 6)
            for spread in FINGER_SPREAD_CANDIDATES_M
        ]
        spread_candidates = sorted(set(spread_candidates + expanded_spreads))
    for frame in frames:
        for mode in GRIPPER_MODES:
            for spread in spread_candidates:
                for rotation_deg in ROTATION_CANDIDATES_DEG:
                    candidate = generate_contacts_for_frame(
                        frame, points, pixels, normals, camera_origin, heatmap,
                        object_principal_axis,
                        point_heat_values=point_heat_values,
                        point_edge_scores=cached_edge_scores,
                        finger_spread_m=spread,
                        gripper_mode=mode,
                        rotation_deg=rotation_deg,
                        object_flexible=object_flexible,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
    return candidates
