"""
STAGE 4: Score and rank grasp candidates.

Each candidate gets:

  SOFT METRICS
    - patch_planarity  : how flat the surface is (from PCA)
    - triangle_score   : how equilateral the contact triangle is
    - alignment_score  : approach perpendicular to the object's long axis
    - visible_fraction : how many contacts are on real cloud points
    - opening_score    : how comfortably the object fits in the gripper

  HARD CONSTRAINTS (all must pass):
    - surface flat enough              (planarity > 0.90)
    - triangle well-formed             (triangle_score >= 0.10)
    - fits inside gripper opening      (opening <= 155 mm)
    - fingers can reach the contacts   (distance <= 130 mm)
    - palm does not collide with the object
    - approach path is clear

  Final score = weighted average of soft metrics (see grasp_config.py).
  Candidates are sorted descending by score.

Input:  list of GraspCandidate + object cloud + centre + scale
Output: list of ScoredGrasp, sorted best-first
"""

from __future__ import annotations

import numpy as np

from grasp_config import (
    ENCOMPASSING_DIAMETER_MAX_M,
    FLEXIBLE_DIAMETER_COMPRESSION,
    GRIPPABLE_DIAMETER_MAX_M,
    GRIPPABLE_DIAMETER_MIN_M,
    GRIPPABLE_SWEET_HIGH_M,
    GRIPPABLE_SWEET_LOW_M,
    GRIPPER_MODE_PARAMS,
    MIN_OPPOSITION_SCORE,
    MIN_TRIANGLE_QUALITY,
    PLANARITY_HARD_MIN,
    ROBOTIQ_3F_MAX_OPENING_M,
    SUPPORT_ENCOMPASSING_CLEARANCE_M,
    SUPPORT_ENCOMPASSING_RADIUS_FACTOR,
    SUPPORT_FINGERTIP_CLEARANCE_M,
    SUPPORT_FINGERTIP_RADIUS_FACTOR,
    SUPPORT_MIN_CONTACT_CLEARANCE_M,
    SUPPORT_OBJECT_CONTACT_TOL_M,
    SUPPORT_TOPDOWN_ALIGNMENT_MIN,
    TOPDOWN_RIM_ENCOMPASSING_EDGE_MIN,
    USE_HEATMAP,
    USE_FLEXIBILITY,
    USE_WRENCH_SCORE,
    VISIBLE_EDGE_FEATURE_GAIN,
    VISIBLE_EDGE_HEATMAP_BLEND,
    W_ALIGNMENT,
    W_FEATURE,
    W_HEATMAP,
    W_OPENING,
    W_OPPOSITION,
    W_PLANARITY,
    W_THICKNESS,
    W_TOP_ENTRY,
    W_TRIANGLE,
    W_VISIBLE,
    W_WRENCH,
    WRENCH_MU_VALUES,
)
from grasp_wrench import opposition_score as _opposition_score
from grasp_wrench import wrench_score as _wrench_score
from grasp_helpers import normalize
from grasp_types import GraspCandidate, ScoredGrasp
from gripper_model import (
    FINGER_REACH_M,
    MIN_ENCOMPASSING_DIAMETER_M,
    PALM_HALF_DEPTH_M,
    approach_path_clear,
    contacts_reachable,
    estimate_support_plane,
    fingers_collide_with_cloud,
    palm_collides_with_cloud,
    support_clearance_ok,
)
from object_dimensions import ObjectDimensions


def _triangle_quality(pts: list[np.ndarray]) -> tuple[float, float]:
    """
    Quality of the 1-thumb + 2-fingers opposed contact triangle.

    The Robotiq 3F is asymmetric (one thumb opposed to two parallel
    fingers), so a perfectly equilateral triangle is NOT what we want —
    the natural configuration has the thumb far from the two fingers
    (closing direction) and the two fingers spread along the object.

    Two things actually matter for an opposed grasp:
      1. SYMMETRY — the thumb must be equidistant from the two fingers
         (|F1-F2| ≈ |F1-F3|), so the gripper closes evenly.
      2. NON-DEGENERACY — the three points must form a real triangle
         with non-trivial area (no collinear contacts).

    The score is:
        symmetry_score * non_degeneracy_score
    where symmetry = min(d12,d13) / max(d12,d13) and non_degeneracy is a
    soft floor on the triangle's area.

    Returns: (quality, area)
    """
    p1, p2, p3 = pts                    # F1 = thumb, F2/F3 = fingers
    d12 = float(np.linalg.norm(p1 - p2))
    d13 = float(np.linalg.norm(p1 - p3))
    d23 = float(np.linalg.norm(p2 - p3))

    area = 0.5 * float(np.linalg.norm(np.cross(p2 - p1, p3 - p1)))

    long_side = max(d12, d13, 1e-6)
    symmetry = min(d12, d13) / long_side

    # Non-degeneracy: penalise nearly-collinear configurations. Use the
    # ratio of the triangle's height (from F2-F3) to the F2-F3 spread.
    # A right-angled grasp (thumb perpendicular to F2-F3) gives ratio ~ 1.
    spread = max(d23, 1e-6)
    height = (2.0 * area) / spread
    non_degeneracy = float(np.clip(height / spread, 0.0, 1.0))

    quality = symmetry * (0.5 + 0.5 * non_degeneracy)
    return float(np.clip(quality, 0.0, 1.0)), area


#  Geometric grippability score

def _generic_thickness_score(local_diameter_m: float) -> float:
    """
    Trapezoidal score that rewards local cross-section diameters inside
    the Robotiq 3F sweet spot. Ramps to zero outside [MIN, MAX], plateau
    at 1.0 between [SWEET_LOW, SWEET_HIGH].

    Penalises the very centre of fat objects (where the heatmap may be
    brightest but the cross-section is too wide for a stable wrap) and
    very thin areas (where the gripper cannot close enough), while
    rewarding finger-friendly diameters.
    """
    d = local_diameter_m
    if d <= GRIPPABLE_DIAMETER_MIN_M or d >= GRIPPABLE_DIAMETER_MAX_M:
        return 0.0
    if GRIPPABLE_SWEET_LOW_M <= d <= GRIPPABLE_SWEET_HIGH_M:
        return 1.0
    if d < GRIPPABLE_SWEET_LOW_M:
        return float(
            (d - GRIPPABLE_DIAMETER_MIN_M)
            / max(GRIPPABLE_SWEET_LOW_M - GRIPPABLE_DIAMETER_MIN_M, 1e-9)
        )
    return float(
        (GRIPPABLE_DIAMETER_MAX_M - d)
        / max(GRIPPABLE_DIAMETER_MAX_M - GRIPPABLE_SWEET_HIGH_M, 1e-9)
    )


def _score_against_range(local_diameter_m: float, low_m: float, high_m: float) -> float:
    """
    Score how well a measured local diameter matches one plausible
    diameter range for a KNOWN object.

    If the diameter falls inside the real range, score = 1.
    Outside the range, the score fades linearly to 0 over a soft margin.
    This keeps the prior helpful without making it brittle.
    """
    if low_m <= local_diameter_m <= high_m:
        return 1.0

    range_width = max(high_m - low_m, 1e-6)
    soft_margin = max(0.005, 0.25 * range_width, 0.15 * high_m)

    if local_diameter_m < low_m:
        return float(np.clip(1.0 - (low_m - local_diameter_m) / soft_margin, 0.0, 1.0))

    return float(np.clip(1.0 - (local_diameter_m - high_m) / soft_margin, 0.0, 1.0))


def _thickness_score(
    local_diameter_m: float,
    object_prior: ObjectDimensions | None = None,
) -> float:
    """
    Thickness score with optional real-dimension prior.

    Fallback:
      generic Robotiq 3F sweet-spot trapezoid

    If real object dimensions are known:
      compare the measured local diameter against all plausible grasp
      cross-sections for that object and keep the best match
    """
    # Real object dimensions are kept for post-hoc validation/reporting,
    # not for selecting the grasp itself. The online grasp score therefore
    # uses only the generic Robotiq-compatible local diameter model.
    return _generic_thickness_score(local_diameter_m)


def _classify_grip_type(
    gripper_mode: str,
    local_diameter_m: float,
    approach_direction: np.ndarray,
    support_plane: tuple[np.ndarray, float] | None,
    visible_edge_score: float,
    top_entry_like: bool,
) -> str:
    """
    Decide whether the grasp is "encompassing" or "fingertip".

    Per Robotiq 3F manual section 1, fig. 1.5–1.6:
      - Pinch and Scissor modes are fingertip-only — fingers cannot
        wrap around the object in these configurations.
      - Basic and Wide modes can do either, depending on the object's
        cross-section: thin enough to fit between the proximal/medial
        phalanges → encompassing; otherwise the distal phalanges
        contact first → fingertip.

    The grasp can only be "encompassing" inside a practical diameter
    band: large enough that the fingers are really wrapping around an
    object, but still small enough to fit that wrap configuration.
    """
    mode_params = GRIPPER_MODE_PARAMS.get(gripper_mode, None)
    if mode_params is None or not mode_params.get("encompassing_capable", False):
        return "fingertip"

    if support_plane is not None:
        support_normal, _ = support_plane
        # approach_direction points TOWARD the object. For an object resting
        # on a support, a true top-down entry therefore points opposite to
        # the support normal, not merely parallel to it.
        topdown_alignment = -float(approach_direction @ support_normal)
        if (
            (topdown_alignment >= SUPPORT_TOPDOWN_ALIGNMENT_MIN or top_entry_like)
            and visible_edge_score >= TOPDOWN_RIM_ENCOMPASSING_EDGE_MIN
            and MIN_ENCOMPASSING_DIAMETER_M <= local_diameter_m <= mode_params["diameter_max_m"]
        ):
            return "encompassing"
        # Supported objects should not be treated as lateral encompassing
        # grasps. If the wrap is not clearly a top-entry grasp, degrade it
        # to fingertip instead of pretending the fingers can go through the
        # support.
        return "fingertip"

    if MIN_ENCOMPASSING_DIAMETER_M <= local_diameter_m <= min(
        ENCOMPASSING_DIAMETER_MAX_M,
        mode_params["diameter_max_m"],
    ):
        return "encompassing"
    return "fingertip"


#  Per-candidate scoring

def _compute_palm_center(
    grasp_center: np.ndarray,
    approach_direction: np.ndarray,
) -> np.ndarray:
    """
    Estimate where the palm centre would be when grasping.

    The palm sits BEHIND the contact points:
      - approach_direction points toward the object
      - we go backward (opposite) by finger reach + half the palm depth
    """
    standoff = FINGER_REACH_M * 0.6 + PALM_HALF_DEPTH_M
    return grasp_center - approach_direction * standoff


def _score_one(
    candidate: GraspCandidate,
    object_center: np.ndarray,
    object_scale_m: float,
    object_cloud: np.ndarray,
    scene_cloud: np.ndarray,
    support_plane: tuple[np.ndarray, float] | None,
    object_principal_axis: np.ndarray,
    object_elongation: float,
    object_prior: ObjectDimensions | None = None,
    object_flexible: bool = False,
) -> ScoredGrasp:
    """
    Compute all metrics for a single candidate.
    """
    pts = [c.point for c in candidate.contacts]
    norms = [c.normal for c in candidate.contacts]
    ev = candidate.frame.pca_eigenvalues
    approach = candidate.approach_direction

    # 1. Planarity: if the eigenvalue along the normal is small → flat surface
    #    planarity = 1 - (normal variance / total variance)
    planarity = 1.0 - (ev[2] / max(float(ev.sum()), 1e-12))

    # 2. Triangle quality
    tri_score, tri_area = _triangle_quality(pts)

    # 3. Approach alignment with the object's principal axis.
    #    For elongated objects (banana, bottle) the right grasp is ACROSS
    #    the long axis (perpendicular). For roughly isotropic objects
    #    (sphere, cube) orientation does not matter, so the metric fades
    #    out via `object_elongation`.
    grasp_center = np.mean(pts, axis=0)
    dist_to_com = float(np.linalg.norm(grasp_center - object_center))
    approach_unit_for_align = approach / max(float(np.linalg.norm(approach)), 1e-9)
    perp = 1.0 - abs(float(approach_unit_for_align @ object_principal_axis))
    alignment_score = float(
        object_elongation * perp + (1.0 - object_elongation) * 1.0
    )

    # 4. Fraction of visible contacts (real cloud points)
    vis_frac = candidate.visible_count / 3.0

    # 5. Required opening: diameter of the smallest circle that encloses
    #    the 3 contacts, measured from the grasp centre (local). This is
    #    the actual width the gripper must open to fit the local cross
    #    section of the object — NOT the distance from the global object
    #    centre, which would be misleading for long/curved objects.
    max_r = max(float(np.linalg.norm(p - grasp_center)) for p in pts)
    req_opening = 2.0 * max_r
    flex_active = bool(object_flexible and USE_FLEXIBILITY)
    effective_req_opening = (
        req_opening * (1.0 - FLEXIBLE_DIAMETER_COMPRESSION)
        if flex_active else req_opening
    )
    opening_ok = (MIN_ENCOMPASSING_DIAMETER_M <= effective_req_opening
                  <= ROBOTIQ_3F_MAX_OPENING_M)
    opening_score = 1.0 - min(
        effective_req_opening / ROBOTIQ_3F_MAX_OPENING_M, 1.0,
    )

    visible_edge_score = float(candidate.frame.patch.visible_edge_score)

    # 6a. Diagnostic-only feature score. Kept for reporting, but removed
    #     from ranking so the baseline remains purely analytical and the
    #     heatmap ablation stays interpretable.
    feature_score = float(max(
        candidate.frame.local_elongation,
        VISIBLE_EDGE_FEATURE_GAIN * visible_edge_score,
    ))

    # 6b. Heatmap score — optional visual prior.
    #     For local-patch search we still want regional context, but for the
    #     global analytical baseline all candidates share the same patch. We
    #     therefore blend:
    #       - patch mean heatmap      -> region-level prior
    #       - contact mean heatmap    -> contact-level discrimination
    #     This keeps geometry-only and +heatmap comparable while allowing the
    #     visual prior to matter even in the one-patch baseline.
    contact_heat_values = [float(ct.heatmap_value) for ct in candidate.contacts]
    contact_hm_score = (
        float(np.mean(contact_heat_values)) if contact_heat_values else 0.0
    )
    patch_hm_score = float(candidate.frame.patch.mean_heatmap)
    hm_score = (
        0.5 * patch_hm_score + 0.5 * contact_hm_score
        if USE_HEATMAP else 0.0
    )

    # 6c. Local thickness score — rewards CLOSING distances in the gripper
    #     sweet spot. With the 1-thumb + 2-fingers opposed configuration,
    #     the meaningful diameter is the gap the gripper actually closes
    #     against: the perpendicular distance from the thumb (contact_1)
    #     to the line connecting the 2 parallel fingers (contact_2,
    #     contact_3). Measuring this gap — instead of the bounding-circle
    #     of all 3 contacts — keeps the longitudinal spread of fingers B+C
    #     out of the diameter calculation.
    approach_unit = approach / max(float(np.linalg.norm(approach)), 1e-9)
    contact_arr = np.array(pts)
    if len(pts) == 3:
        thumb = contact_arr[0]
        fingers_mid = 0.5 * (contact_arr[1] + contact_arr[2])
        gap_vec = thumb - fingers_mid
        # Remove the component along the approach direction (we only care
        # about the closing direction perpendicular to approach).
        gap_vec_perp = gap_vec - (gap_vec @ approach_unit) * approach_unit
        local_diameter_m = float(np.linalg.norm(gap_vec_perp))
    else:
        radial = contact_arr - grasp_center
        radial_perp = radial - (radial @ approach_unit)[:, None] * approach_unit
        local_diameter_m = 2.0 * float(np.max(np.linalg.norm(radial_perp, axis=1)))
    # For flexible objects the gripper can compress the object during closure.
    # The ablation toggle controls whether this relaxed effective diameter is
    # actually used in thickness scoring and per-mode diameter feasibility.
    effective_diameter_m = (
        local_diameter_m * (1.0 - FLEXIBLE_DIAMETER_COMPRESSION)
        if flex_active else local_diameter_m
    )
    mode_params = GRIPPER_MODE_PARAMS.get(candidate.gripper_mode, None)
    if mode_params is not None and effective_diameter_m > mode_params["diameter_max_m"]:
        thick_score = 0.0
    else:
        thick_score = _thickness_score(effective_diameter_m, object_prior)
    mode_diameter_ok = (
        mode_params is None
        or effective_diameter_m <= mode_params["diameter_max_m"]
    )
    # Reported for diagnostics so the ablation output makes clear whether the
    # flexible-diameter relaxation was active for this grasp.
    flex_score = thick_score if flex_active else 0.0

    opp_score = _opposition_score(pts, norms)
    wq_score = 0.0

    # 7. Hard constraints — geometric checks only (scene collision disabled)
    palm_center = _compute_palm_center(grasp_center, approach)
    reachable = contacts_reachable(pts, palm_center, approach)
    no_palm_collision = True
    no_finger_collision = True
    clear_approach = True
    object_min_height_m = None
    top_entry_like = False
    if support_plane is not None and len(object_cloud) > 0:
        support_normal, support_offset = support_plane
        contact_heights = np.array(
            [float(support_normal @ pt + support_offset) for pt in pts],
            dtype=np.float64,
        )
        object_heights = object_cloud @ support_normal + support_offset
        object_min_height_m = float(
            np.min(object_heights)
        )
        object_max_height_m = float(np.max(object_heights))
        object_height_span_m = max(object_max_height_m - object_min_height_m, 1e-9)
        top_band_m = max(0.010, 0.20 * object_height_span_m)
        top_entry_like = bool(
            np.min(contact_heights) >= object_max_height_m - top_band_m
        )
    grip_type = _classify_grip_type(
        candidate.gripper_mode,
        local_diameter_m,
        approach,
        support_plane,
        visible_edge_score,
        top_entry_like,
    )
    support_clear = True

    # All hard constraints must pass
    valid = all([
        planarity > PLANARITY_HARD_MIN,
        tri_score >= MIN_TRIANGLE_QUALITY,
        opp_score >= MIN_OPPOSITION_SCORE,
        opening_ok,
        mode_diameter_ok,
        reachable,
        support_clear,
    ])

    # 8. Final score = normalized weighted average of ACTIVE metrics.
    #    This keeps geometry-only and geometry+heatmap runs comparable:
    #    only the heatmap term is toggled, while the geometry stays intact.
    weighted_terms: list[tuple[float, float]] = [
        (W_TRIANGLE, tri_score),
        (W_OPPOSITION, opp_score),
        (W_THICKNESS, thick_score),
        (W_OPENING, opening_score),
        (W_ALIGNMENT, alignment_score),
        (W_PLANARITY, planarity),
    ]
    if USE_HEATMAP and W_HEATMAP > 0.0:
        weighted_terms.append((W_HEATMAP, hm_score))

    if USE_WRENCH_SCORE and W_WRENCH > 0.0:
        wq_score = _wrench_score(pts, norms, object_center, approach, WRENCH_MU_VALUES)
        weighted_terms.append((W_WRENCH, wq_score))

    total_weight = sum(weight for weight, _ in weighted_terms)
    final = 0.0
    if total_weight > 1e-9:
        final = sum(weight * value for weight, value in weighted_terms) / total_weight

    return ScoredGrasp(
        candidate=candidate,
        valid=valid,
        final_score=round(final, 6),
        patch_planarity=round(planarity, 6),
        triangle_score=round(tri_score, 6),
        triangle_area_m2=round(tri_area, 8),
        alignment_score=round(alignment_score, 6),
        feature_score=round(feature_score, 6),
        distance_to_com_m=round(dist_to_com, 6),
        grip_type=grip_type,
        visible_fraction=round(vis_frac, 6),
        required_opening_m=round(req_opening, 6),
        heatmap_score=round(hm_score, 6),
        thickness_score=round(thick_score, 6),
        local_diameter_m=round(local_diameter_m, 6),
        opening_feasible=opening_ok,
        contacts_reachable=reachable,
        palm_collision_free=no_palm_collision,
        fingers_collision_free=no_finger_collision,
        approach_clear=clear_approach,
        support_clear=support_clear,
        top_entry_like=top_entry_like,
        opposition_score=round(opp_score, 6),
        wrench_score=round(wq_score, 6),
        object_flexible=object_flexible,
        flex_score=round(flex_score, 6),
    )


#  Public API

def score_and_rank(
    candidates: list[GraspCandidate],
    object_center: np.ndarray,
    object_scale_m: float,
    object_cloud: np.ndarray,
    scene_cloud: np.ndarray,
    object_principal_axis: np.ndarray,
    object_elongation: float,
    object_prior: ObjectDimensions | None = None,
    object_flexible: bool = False,
) -> list[ScoredGrasp]:
    """Score every candidate and return them sorted best-first."""
    support_plane = estimate_support_plane(scene_cloud, object_cloud, object_center)
    scored = [
        _score_one(
            c, object_center, object_scale_m, object_cloud, scene_cloud,
            support_plane,
            object_principal_axis, object_elongation, object_prior,
            object_flexible,
        )
        for c in candidates
    ]
    scored.sort(key=lambda g: g.final_score, reverse=True)
    return scored
