"""
Physical model of the Robotiq 3-Finger Adaptive Gripper.

All values from the official manual: 3-Finger_PDF_20190221.pdf

This module contains:
  1. Real gripper dimensions (from the manual)
  2. Collision check functions (palm and approach path)
  3. Finger reachability check

Lengths in metres, forces in Newtons, angles in radians.
"""

from __future__ import annotations

import numpy as np


# ===================================================================
#  Main dimensions (Section 6.1, Figure 6.1.1)
# ===================================================================

GRIPPER_HEIGHT_M = 0.130              # 130 mm — gripper height
GRIPPER_WIDTH_M = 0.204               # 204 mm — width
GRIPPER_DEPTH_M = 0.092               # 92 mm  — depth
GRIPPER_PALM_DIAMETER_M = 0.080       # 80 mm  — palm diameter
BASIC_WIDE_PINCH_OPENING_M = 0.155    # 155 mm — opening in Wide mode


# ===================================================================
#  Mechanical specs (Section 6.2)
# ===================================================================

GRIPPER_OPENING_MAX_M = 0.167         # 167 mm — max opening (custom pads)
MAX_ENCOMPASSING_DIAMETER_M = 0.155   # 155 mm — largest object that fits
MIN_ENCOMPASSING_DIAMETER_M = 0.020   # 20 mm  — smallest practical object
GRIPPER_WEIGHT_KG = 2.3

PAYLOAD_ENCOMPASSING_KG = 10.0        # 10 kg  — payload in encompassing mode
PAYLOAD_FINGERTIP_KG = 2.5            # 2.5 kg — payload in fingertip mode

GRIP_FORCE_MAX_N = 70.0               # 70 N   — max fingertip force
GRIP_FORCE_MIN_N = 15.0               # 15 N   — min force
CLOSING_SPEED_MAX_M_S = 0.110         # 110 mm/s — max closing speed


# ===================================================================
#  Detailed dimensions (pads, fingertips — Sections 6.3–6.5)
# ===================================================================

FINGERTIP_LENGTH_M = 0.0307           # 30.7 mm
FINGERTIP_WIDTH_M = 0.0102            # 10.2 mm
PALM_PAD_WIDTH_M = 0.0635             # 63.5 mm
SUPPLY_VOLTAGE_V = 24.0
PEAK_POWER_W = 36.0


# ===================================================================
#  Centre of mass and inertia (Section 6.4)
# ===================================================================

CENTER_OF_MASS_M = np.array([-0.008, 0.0, 0.065])
INERTIA_KG_MM2 = np.array([
    [7300.0,    0.0, -650.0],
    [   0.0, 8800.0,    0.0],
    [-650.0,    0.0, 7000.0],
])


# ===================================================================
#  Derived values for collision checks
# ===================================================================

PALM_RADIUS_M = GRIPPER_PALM_DIAMETER_M / 2.0   # 40 mm
PALM_HALF_DEPTH_M = GRIPPER_DEPTH_M / 2.0       # 46 mm
FINGER_REACH_M = GRIPPER_HEIGHT_M                # 130 mm (palm to fingertip)


# ===================================================================
#  Collision checks
#
#  The gripper is modelled as a simplified shape:
#    - palm = cylinder (radius 40mm, depth 92mm)
#    - fingers = reach of 130mm from the palm face
#    - approach path = cylinder behind the palm
# ===================================================================

def palm_collides_with_cloud(
    palm_center: np.ndarray,
    approach_direction: np.ndarray,
    cloud_points: np.ndarray,
    safety_margin_m: float = 0.005,
) -> bool:
    """
    Check if the palm cylinder would hit any cloud points.

    The palm is modelled as a cylinder along the approach axis.
    If any cloud point is INSIDE the cylinder → collision.

    Steps:
      1. Compute how far each point is along the approach axis (along).
      2. Compute how far each point is perpendicular to the axis (perp).
      3. If a point is within both the cylinder length and radius → collision.
    """
    radius = PALM_RADIUS_M + safety_margin_m

    offsets = cloud_points - palm_center
    along = offsets @ approach_direction                       # dist along axis
    perp = offsets - np.outer(along, approach_direction)       # perpendicular part
    perp_dist = np.linalg.norm(perp, axis=1)

    inside = (np.abs(along) < PALM_HALF_DEPTH_M) & (perp_dist < radius)
    return bool(np.any(inside))


def approach_path_clear(
    palm_center: np.ndarray,
    approach_direction: np.ndarray,
    cloud_points: np.ndarray,
    path_length_m: float = 0.10,
    clearance_radius_m: float = 0.045,
) -> bool:
    """
    Check that the approach path (behind the palm) is free of obstacles.

    The gripper arrives from behind (opposite to the approach direction).
    If there are cloud points on this path → the gripper cannot reach.

    Steps:
      1. Compute distance along the approach axis (behind = negative).
      2. Filter points on the path (between 0 and -path_length_m).
      3. Check that all of them are far enough from the axis.
    """
    offsets = cloud_points - palm_center
    along = offsets @ approach_direction

    # "Behind the palm" = negative projection along the approach direction
    on_path = (along < 0) & (along > -path_length_m)

    if not np.any(on_path):
        return True   # no points on the path — all clear

    perp = offsets[on_path] - np.outer(along[on_path], approach_direction)
    perp_dist = np.linalg.norm(perp, axis=1)

    return bool(np.all(perp_dist > clearance_radius_m))


def contacts_reachable(
    contact_points: list[np.ndarray],
    palm_center: np.ndarray,
    approach_direction: np.ndarray | None = None,
) -> bool:
    """
    Check that each contact is within finger reach from the palm.

    Fingers have a finite length (FINGER_REACH_M = 130mm).
    If a contact point is farther than the fingers can reach → invalid.
    """
    if approach_direction is not None:
        # The palm face is half the depth forward along the approach direction
        palm_face = palm_center + approach_direction * PALM_HALF_DEPTH_M
    else:
        palm_face = palm_center

    for point in contact_points:
        dist = float(np.linalg.norm(point - palm_face))
        if dist > FINGER_REACH_M:
            return False
    return True


def fingers_collide_with_cloud(
    contact_points: list[np.ndarray],
    palm_center: np.ndarray,
    approach_direction: np.ndarray,
    cloud_points: np.ndarray,
    finger_thickness_m: float = 0.012,
    proximal_check_fraction: float = 0.35,
) -> bool:
    """
    Check that no part of the fingers passes through the object.

    The Robotiq 3F has CURVED, articulated fingers — they extend forward
    along the approach direction and then curl in around the object.
    Modelling each finger as a straight cylinder from the palm face to
    its contact point would mark every opposed grasp as colliding (the
    straight line crosses the object). To approximate the curl, we only
    flag a collision when a cloud point sits inside the PROXIMAL part of
    the finger (the rigid section attached to the palm). The distal
    section curls and is not modelled — clearance there is enforced
    instead by the palm and approach-path checks.
    """
    palm_face = palm_center + approach_direction * PALM_HALF_DEPTH_M

    for contact in contact_points:
        finger_axis = contact - palm_face
        finger_length = float(np.linalg.norm(finger_axis))
        if finger_length < 1e-6:
            continue
        finger_dir = finger_axis / finger_length

        # Only check the proximal section — the rigid base of the finger.
        proximal_end = finger_length * proximal_check_fraction

        offsets = cloud_points - palm_face
        along = offsets @ finger_dir
        on_finger = (along > 0.0) & (along < proximal_end)
        if not np.any(on_finger):
            continue

        perp = offsets[on_finger] - np.outer(along[on_finger], finger_dir)
        perp_dist = np.linalg.norm(perp, axis=1)
        if np.any(perp_dist < finger_thickness_m):
            return True
    return False


def estimate_support_plane(
    scene_points: np.ndarray,
    object_points: np.ndarray,
    object_center: np.ndarray,
    inlier_tolerance_m: float = 0.008,
    object_clearance_m: float = 0.010,
    neighborhood_scale: float = 1.8,
    max_points: int = 3000,
    ransac_iters: int = 120,
) -> tuple[np.ndarray, float] | None:
    """
    Estimate the dominant support plane from the observed scene cloud.

    Returns a plane in Hessian form:
      normal . x + offset = signed_distance

    The normal is oriented so the object centre lies on the positive side,
    i.e. "above" the support.
    """
    if len(scene_points) < 50:
        return None

    object_bbox = object_points.max(axis=0) - object_points.min(axis=0)
    object_radius = max(float(np.linalg.norm(object_bbox)) * 0.5, 0.03)
    neighborhood_radius = max(neighborhood_scale * object_radius, 0.08)

    # Keep only scene points near the object. A global scene cloud often
    # contains walls or distant surfaces that dominate RANSAC even though
    # they are irrelevant to the local support/contact situation.
    center_dist = np.linalg.norm(scene_points - object_center, axis=1)
    local_scene = scene_points[center_dist <= neighborhood_radius]
    if len(local_scene) < 50:
        local_scene = scene_points

    # Remove points that belong to the object itself (or lie extremely close
    # to it). The support plane should come from neighbours / table, not from
    # the visible object surface.
    if len(object_points) > 0 and len(local_scene) > 0:
        min_dist2 = np.full(len(local_scene), np.inf, dtype=np.float64)
        chunk = 512
        for start in range(0, len(object_points), chunk):
            obj_block = object_points[start:start + chunk]
            diff = local_scene[:, None, :] - obj_block[None, :, :]
            dist2 = np.einsum("ijk,ijk->ij", diff, diff)
            min_dist2 = np.minimum(min_dist2, dist2.min(axis=1))
        keep = min_dist2 > (object_clearance_m ** 2)
        points = local_scene[keep]
    else:
        points = local_scene

    if len(points) < 50:
        points = local_scene
    if len(points) < 50:
        return None

    if len(points) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]

    best_inliers = None
    best_count = 0
    rng = np.random.default_rng(1)

    for _ in range(ransac_iters):
        sample_idx = rng.choice(len(points), size=3, replace=False)
        p0, p1, p2 = points[sample_idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -float(normal @ p0)

        signed_object = float(normal @ object_center + offset)
        if signed_object < 0.0:
            normal = -normal
            offset = -offset
            signed_object = -signed_object
        if signed_object < inlier_tolerance_m:
            continue

        distances = np.abs(points @ normal + offset)
        inliers = distances < inlier_tolerance_m
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = points[inliers]

    def _fit_plane_pca(plane_points: np.ndarray) -> tuple[np.ndarray, float] | None:
        if len(plane_points) < 30:
            return None
        centroid = plane_points.mean(axis=0)
        centred = plane_points - centroid
        cov = (centred.T @ centred) / max(len(plane_points) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, int(np.argmin(eigvals))]
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        offset = -float(normal @ centroid)
        if float(normal @ object_center + offset) < 0.0:
            normal = -normal
            offset = -offset
        signed_object = float(normal @ object_center + offset)
        if signed_object < inlier_tolerance_m:
            return None
        return normal, offset

    if best_inliers is None or len(best_inliers) < 30:
        # Fallback: assume the object rests on a support surface.
        # If local RANSAC is inconclusive, approximate that local support with
        # a PCA plane fitted on the nearby scene points rather than treating
        # the object as floating in space.
        return _fit_plane_pca(points)

    centroid = best_inliers.mean(axis=0)
    centred = best_inliers - centroid
    cov = (centred.T @ centred) / max(len(best_inliers) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, int(np.argmin(eigvals))]
    normal /= max(float(np.linalg.norm(normal)), 1e-9)
    offset = -float(normal @ centroid)
    if float(normal @ object_center + offset) < 0.0:
        normal = -normal
        offset = -offset
    signed_object = float(normal @ object_center + offset)
    if signed_object < inlier_tolerance_m:
        return _fit_plane_pca(points)
    return normal, offset


def support_clearance_ok(
    grasp_center: np.ndarray,
    contact_points: list[np.ndarray],
    approach_direction: np.ndarray,
    local_diameter_m: float,
    grip_type: str,
    support_plane: tuple[np.ndarray, float] | None,
    object_min_height_m: float | None,
    top_entry_like: bool,
    encompassing_radius_factor: float,
    fingertip_radius_factor: float,
    encompassing_clearance_m: float,
    fingertip_clearance_m: float,
    min_contact_clearance_m: float,
    object_contact_tol_m: float,
    topdown_alignment_min: float,
) -> bool:
    """
    Reject grasps that would need to pass through the support surface.

    Side grasps need the centre of the grasp to sit sufficiently above the
    support plane so the lower part of the finger wrap does not intersect the
    table. Top-down grasps are exempt from that stronger rule.
    """
    if support_plane is None:
        return True

    normal, offset = support_plane
    contact_heights = np.array([float(normal @ pt + offset) for pt in contact_points])
    if float(np.min(contact_heights)) < min_contact_clearance_m:
        return False

    # The grasp approaches TOWARD the object. "Coming from above" means the
    # approach direction points opposite to the support normal. Using abs(...)
    # would incorrectly also accept an approach coming from below the object.
    vertical_alignment = -float(approach_direction @ normal)
    if vertical_alignment >= topdown_alignment_min:
        return True

    if grip_type == "encompassing" and top_entry_like:
        return True

    # Conservative rule for supported scenes: if a support plane is present
    # and the grasp is not top-down, do not allow an encompassing wrap.
    # With only the observed surface available, a lateral wrap would require
    # the lower part of the fingers to pass through whatever the object rests
    # on, so we reject it generically.
    object_resting = (
        object_min_height_m is not None
        and object_min_height_m <= object_contact_tol_m
    )
    if grip_type == "encompassing" and (object_resting or support_plane is not None):
        return False

    if grip_type == "encompassing":
        required_height = (
            encompassing_radius_factor * local_diameter_m
            + encompassing_clearance_m
        )
    else:
        required_height = (
            fingertip_radius_factor * local_diameter_m
            + fingertip_clearance_m
        )

    grasp_height = float(normal @ grasp_center + offset)
    return grasp_height >= required_height
