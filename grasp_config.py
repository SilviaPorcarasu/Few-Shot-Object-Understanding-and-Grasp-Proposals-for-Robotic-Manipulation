"""
Constants for the Robotiq 3-Finger grasp generation pipeline.

All values are in meters / dimensionless.
Adjust parameters here without touching the logic in other files.
"""

import os
from pathlib import Path

# ── Data paths ──────────────────────────────────────────────────────
# The pipeline reads data from a "frame" produced by upstream:
#   - <frame>_grasp_patch_data.npz  (points, pixels, normals, camera_origin)
#   - <frame>_object_cloud.ply      (full object for collision checks)
#   - <frame>_grasp_patch_cloud.ply (patch with normals — same points as npz)
#   - <frame>_combined_grasp_cloud.ply (symmetric case: includes mirrored points)
# And the heatmap from the segmentation stage:
#   - segmentation/sam_inference/<frame_name>_pred_gray.png

ROOT = Path(__file__).resolve().parent

# Search strategy:
#   - global_pca_baseline: one global frame on the whole object; simplest
#     analytical baseline, useful before adding visual priors
#   - local_patches: farthest-point sampled local search (the richer variant)
GRASP_SEARCH_STRATEGY = os.environ.get(
    "GRASP_SEARCH_STRATEGY", "global_pca_baseline"
).strip().lower()

# Default frame to process — change here to switch object/case.
# The batch runner updates this path for each grasp-ready object.
FRAME_DIR = (ROOT.parent / 'refactored_sam_pipeline-3/grasp_ready_objects/lotion_bottle/primary_symmetric_3d')
FRAME_PREFIX = "lotion_bottle"

# Active object from the catalog below (None = use generic parameters).
# Set this to a key from OBJECT_CATALOG to auto-tune gripper params for
# a specific real object.  Example: ACTIVE_OBJECT = "cana"
ACTIVE_OBJECT: str | None = None

# Heatmap directory (from upstream segmentation pipeline)
HEATMAP_DIR = (ROOT.parent / 'refactored_sam_pipeline-3/grasp_ready_heatmaps')


# ── Patch sampling ──────────────────────────────────────────────────
# select_patches.py uses these to split the grasp zone into smaller sub-zones.

SEED_COUNT = 24               # how many seed points (farthest-point sampling)
PATCH_RADIUS_M = 0.014        # 14 mm — sphere radius around each seed
PATCH_MIN_POINTS = 200         # patches with fewer points are discarded

# Threshold on the projected heatmap when picking object points to consider
# as grasp candidates. Set to 0.0 to disable heatmap filtering and search
# the whole object cloud purely on geometric criteria.
HEATMAP_FILTER_THRESHOLD = 0.0   # 0..1   (0 = disabled)
USE_HEATMAP: bool = os.environ.get("USE_HEATMAP", "1").strip().lower() not in ("0", "false", "no")
USE_FLEXIBILITY: bool = os.environ.get("USE_FLEXIBILITY", "1").strip().lower() not in ("0", "false", "no")
HEATMAP_LOCAL_PATCH_TOPK = 4      # when heatmap is active, also try the top-K local hot patches
HEATMAP_LOCAL_PATCH_MIN = 0.10    # ignore weak local patches below this mean heat
HEATMAP_CONTACT_BIAS = 0.25       # how much heatmap influences the exact F1/F2/F3 choice
FLEXIBLE_SPREAD_SCALE = float(os.environ.get("FLEXIBLE_SPREAD_SCALE", "1.15"))
FLEXIBLE_LONG_TOLERANCE_SCALE = float(os.environ.get("FLEXIBLE_LONG_TOLERANCE_SCALE", "1.25"))
INTRINSICS_PATH_NAME = "camera_intrinsics.json"
RGB_W = 1280
RGB_H = 720

# Kept only for reporting/visual diagnostics. The baseline ranking uses the
# raw heatmap score directly so the comparison geometry-only vs +heatmap
# stays clean and interpretable.
VISIBLE_EDGE_HEATMAP_BLEND = 0.0
VISIBLE_EDGE_FEATURE_GAIN = 0.0


# ── Scoring ─────────────────────────────────────────────────────────
# score_grasps.py uses these values.

ROBOTIQ_3F_MAX_OPENING_M = 0.155   # 155 mm — max gripper opening
MIN_TRIANGLE_QUALITY = 0.10        # contact triangle must be sufficiently equilateral
                                   # (0 = degenerate, 1 = perfect equilateral)

# ── Score weights ──────────────────────────────────────────────────
# We keep a clean analytical baseline:
#   geometry-only: triangle + opposition + diameter/opening + alignment + planarity
#   geometry+heatmap: same candidate set and same geometric score, plus heatmap
# The final score is normalized over the ACTIVE weights, so turning the
# heatmap off does not unfairly shrink all scores.
W_HEATMAP    = float(os.environ.get("W_HEATMAP", "0.18"))   # optional visual prior, only active when USE_HEATMAP=1
W_THICKNESS  = float(os.environ.get("W_THICKNESS", "0.22"))   # local closing distance is in the gripper sweet-spot
W_TRIANGLE   = float(os.environ.get("W_TRIANGLE", "0.24"))   # approximately isosceles + non-degenerate triangle
W_ALIGNMENT  = float(os.environ.get("W_ALIGNMENT", "0.12"))   # approach makes sense relative to object geometry
W_FEATURE    = 0.00   # disabled in ranking; kept only for diagnostics
W_TOP_ENTRY  = 0.00   # disabled in ranking; support handling stays geometric
W_PLANARITY  = float(os.environ.get("W_PLANARITY", "0.08"))   # local patch is stable enough for contact placement
W_OPENING    = float(os.environ.get("W_OPENING", "0.14"))   # gripper opening is comfortable, not near the limit
W_VISIBLE    = 0.00   # disabled in ranking; kept only for diagnostics

# ── Wrench-space quality ────────────────────────────────────────────
# Two variants from grasp_wrench.py:
#
#   USE_WRENCH_SCORE = False  → use opposition_score only (geometric,
#                               friction-free, fast).
#   USE_WRENCH_SCORE = True   → use full friction-cone wrench_score
#                               (NNLS per perturbation, ~3× slower).
#
# W_OPPOSITION and W_WRENCH are additive to the weights above.
# Set both to 0.0 to disable wrench metrics entirely.
# Override from shell: WRENCH=1 python3 run_grasps_on_sam3_outputs.py
USE_WRENCH_SCORE: bool = os.environ.get("WRENCH", "0").strip().lower() in ("1", "true", "yes")

W_OPPOSITION = float(os.environ.get("W_OPPOSITION", "0.20"))   # normal-opposition geometric score (analytical baseline)
W_WRENCH     = 0.00   # wrench-space quality score (friction-cone, disabled by default)
# Flexibility no longer acts as an extra bonus term. Instead, when enabled,
# it relaxes the effective local diameter used by thickness feasibility and
# scoring. The environment flag above is kept so we can run clean ablations
# between heatmap-only and heatmap+flexibility.
W_FLEXIBLE   = float(os.environ.get("W_FLEXIBLE", "0.10"))

# Opposition should matter not only in the score, but also as a minimum
# validity check so obviously non-opposed grasps are filtered out early.
MIN_OPPOSITION_SCORE = 0.20

# μ values to test (rubber fingertip on unknown surface).
# wrench_score returns the minimum over these values (conservative).
WRENCH_MU_VALUES: tuple[float, ...] = (0.2, 0.4, 0.6)

# Hard constraint threshold on local patch planarity. 0.70 admits gently
# curved regions (sides of a banana, sides of a bottle) while still
# rejecting sharp tips and noisy patches.
PLANARITY_HARD_MIN = 0.70

# Support/table constraint. When an object rests on a support surface, a
# side grasp must leave enough room for the lower part of the gripper.
# Otherwise an "encompassing" wrap may look valid in the cloud but would
# require the fingers to go through the table.
SUPPORT_ENCOMPASSING_RADIUS_FACTOR = 0.40
SUPPORT_FINGERTIP_RADIUS_FACTOR = 0.15
SUPPORT_ENCOMPASSING_CLEARANCE_M = 0.002
SUPPORT_FINGERTIP_CLEARANCE_M = 0.001
SUPPORT_MIN_CONTACT_CLEARANCE_M = -0.002
SUPPORT_OBJECT_CONTACT_TOL_M = 0.008
SUPPORT_TOPDOWN_ALIGNMENT_MIN = 0.75
TOPDOWN_RIM_ENCOMPASSING_EDGE_MIN = 0.45

# Sweet-spot range for the local closing distance (m).
# Bounded by realistic object thickness, not the gripper's max opening.
# Real-world thicknesses: banana ~30 mm, telecomandă ~25 mm, sticlă
# ~60-70 mm, mug ~80 mm. A measured "diameter" much above ~70 mm on a
# banana usually means the 1+2 opposed contact placement has misfired
# (F1 on the far side of a curve, etc.) — better to penalise it than to
# call it valid.
GRIPPABLE_DIAMETER_MIN_M = 0.020   # 20 mm
GRIPPABLE_DIAMETER_MAX_M = 0.070   # 70 mm
GRIPPABLE_SWEET_LOW_M    = 0.025   # 25 mm — start of plateau
GRIPPABLE_SWEET_HIGH_M   = 0.050   # 50 mm — end of plateau


# For flexible/deformable objects the gripper can compress the object slightly.
# This factor relaxes per-mode diameter_max_m by the given fraction (0.15 = 15%).
FLEXIBLE_DIAMETER_COMPRESSION = float(os.environ.get("FLEXIBLE_DIAMETER_COMPRESSION", "0.15"))


# ── Contact placement (1-thumb + 2-fingers opposed) ────────────────
# Spread between the two parallel fingers (B+C) measured along the
# object's principal axis. The thumb (A) sits opposite, at the centre
# of the spread. For elongated objects this distributes the contacts
# along the length instead of clustering them on one cross-section.
FINGER_SPREAD_LONG_M = 0.050         # default 50 mm between the 2 parallel fingers
CONTACT_LONG_TOLERANCE_M = 0.015     # ±15 mm window when searching cloud
CONTACT_PLANE_HALF_THICKNESS_M = 0.045  # ±45 mm of patch plane = local search depth

# Fraction of the object's height to exclude from the bottom when placing
# contacts. We define "up" on a fixed camera-frame axis so the constraint is
# stable across runs and does not drift with the current view direction.
# The RGB-D camera frame used here follows the common convention X right,
# Y down, Z forward, so "up" is approximately -Y.
CONTACT_BOTTOM_EXCLUSION_FRACTION = 0.15
CONTACT_UP_AXIS_CAMERA = (0.0, -1.0, 0.0)

# A list of finger spreads to try per patch: pipeline picks whichever
# variant scores best. This auto-adapts the gripper aperture to small
# features (cup handle, narrow neck) vs larger bodies (banana, bottle).
FINGER_SPREAD_CANDIDATES_M = (0.025, 0.050, 0.075)

# Keep the grasp conservative: the object is reconstructed only from the
# observed surface / fused visible patches, not from a complete CAD-like 3D
# model. For the main ranking we therefore keep only the neutral gripper
# rotation and do not let aggressive in-plane rotations dominate selection.
ROTATION_CANDIDATES_DEG = (0,)


# ── Robotiq 3F Operation Modes (manual section 1, fig. 1.3) ────────
# The 3F has four distinct grip modes. They differ in how fingers B
# and C are oriented relative to thumb (finger A) and what kind of
# grasp is mechanically possible:
#   BASIC      — B and C parallel, opposite A. Encompassing or fingertip.
#                Best for "one dimension longer than the other two"
#                (banana, bottle, telecomandă).
#   WIDE       — B and C splayed outward (~32°). Encompassing or fingertip.
#                Best for round / large objects (mug body, ball, bowl).
#   PINCH      — All three fingertips converge to a small point.
#                Fingertip-only, no encompass. Best for small precise
#                pickups (handle of a mug, screw head, cable).
#   SCISSOR    — B and C close laterally toward each other, A stays
#                still. Fingertip-only. Best for tiny / thin objects
#                that fit between B and C only (a coin, a card edge).
#
# Each mode shapes the geometric placement strategy: the perpendicular
# offsets of B vs C, the longitudinal spread, and which configuration
# is even reachable on a given patch. The pipeline tries every mode
# per frame and lets scoring choose. This is what makes the system
# generalise across a banana, a mug (handle in PINCH, body in WIDE),
# a sphere (WIDE encompassing), or a thin card (SCISSOR).
GRIPPER_MODE_BASIC   = "basic"
GRIPPER_MODE_WIDE    = "wide"
GRIPPER_MODE_PINCH   = "pinch"
GRIPPER_MODE_SCISSOR = "scissor"
GRIPPER_MODES = (
    GRIPPER_MODE_BASIC,
    GRIPPER_MODE_WIDE,
    GRIPPER_MODE_PINCH,
    GRIPPER_MODE_SCISSOR,
)

# Per-mode geometric parameters. Each entry tunes how the 3 contacts
# are placed in the patch-local frame:
#   long_spread_scale  — multiplier on FINGER_SPREAD_LONG_M for the
#                        B-C separation along the object's long axis.
#                        Pinch/Scissor barely spread; Wide spreads more.
#   perp_offset_b_c    — extra perpendicular splay of B and C beyond
#                        the rim (Wide pushes them outward; Scissor
#                        pulls them inward toward A's side).
#   thumb_offset_scale — how far the thumb (A) sits from the centre of
#                        B-C, in units of the closing distance. Pinch
#                        brings A close to B-C; Wide keeps it further.
#   diameter_max_m     — maximum closing diameter accepted in this
#                        mode (Pinch and Scissor are fingertip-only,
#                        so they fail on thick objects).
GRIPPER_MODE_PARAMS = {
    GRIPPER_MODE_BASIC: {
        "long_spread_scale":  1.0,
        "perp_offset_b_c":    0.000,
        "thumb_offset_scale": 1.0,
        "diameter_max_m":     0.070,
        # Encompassing only available in Basic/Wide (manual fig. 1.6).
        "encompassing_capable": True,
    },
    GRIPPER_MODE_WIDE: {
        "long_spread_scale":  1.4,
        "perp_offset_b_c":    0.010,
        "thumb_offset_scale": 1.2,
        "diameter_max_m":     0.110,
        "encompassing_capable": True,
    },
    GRIPPER_MODE_PINCH: {
        "long_spread_scale":  0.3,
        "perp_offset_b_c":    0.000,
        "thumb_offset_scale": 0.7,
        "diameter_max_m":     0.030,
        # Pinch is fingertip-only (manual fig. 1.6).
        "encompassing_capable": False,
    },
    GRIPPER_MODE_SCISSOR: {
        "long_spread_scale":  0.0,
        "perp_offset_b_c":   -0.005,
        "thumb_offset_scale": 0.0,
        "diameter_max_m":     0.025,
        # Scissor is fingertip-only (manual fig. 1.6).
        "encompassing_capable": False,
    },
}

# Threshold (m) below which an object is considered "encompassable"
# in Basic/Wide modes — fingers can wrap around it. Above this, the
# grasp degrades to fingertip even in Basic/Wide. Value chosen so a
# typical bottle/banana (Ø ~30-50mm) encompasses, while a thick mug
# body (Ø ~80mm) stays fingertip.
ENCOMPASSING_DIAMETER_MAX_M = 0.060

# When the patch is locally elongated (handle, neck, ridge) we use the
# local PCA axis as `tangent_long` instead of the global object axis —
# the gripper should align with the LOCAL feature, not the whole
# object. Threshold below which the patch is considered isotropic.
LOCAL_ELONGATION_THRESHOLD = 0.40

# Rim-percentile: when picking each contact, take only the top 10% most
# extreme points on the closing axis (rim of the visible surface),
# instead of the absolute argmax. This pushes every finger onto an
# edge/silhouette of the object, not onto the smooth middle.
CONTACT_RIM_FRACTION = 0.10          # keep top 10% on the closing direction (rigid)
CONTACT_RIM_FRACTION_FLEXIBLE = 0.25 # wider band for flexible objects — fingers press into material


# ── Object Catalog ──────────────────────────────────────────────────
# Real physical dimensions and per-object tuned gripper parameters.
#
# Keys match FRAME_PREFIX values set by run_grasps_on_sam3_outputs.py
# (e.g. "blue_cup", "roll_of_tape", "screwdriver", "computer_mouse").
# When ACTIVE_OBJECT equals one of these keys the generic parameters
# above are replaced at import time by the object-specific ones.
#
# Dimension source (all measures in metres):
#   Cană/Blue Cup  — Ø85 mm top, Ø65 mm base, h 120 mm
#   Rolă           — Ø100 mm outer, h 50 mm
#   Șurubelniță    — l 205 mm, handle Ø20 mm, shaft Ø5 mm
#   Mouse          — l 100 mm, w 60 mm, h 30 mm

OBJECT_CATALOG: dict[str, dict] = {

    # ── Cană / Blue Cup ─────────────────────────────────────────────
    # Body Ø85 mm exceeds BASIC mode cap (70 mm) → needs WIDE mode.
    # Handle grip ≈ 15-20 mm → PINCH mode.
    # Sweet-spot plateau spans both zones so neither is penalised by
    # the trapezoidal thickness score.
    "blue_cup": {
        "display_name": "Cană / Blue Cup",
        "grippable_diameter_min_m":    0.015,   # 15 mm (thin handle interior)
        "grippable_diameter_max_m":    0.095,   # 95 mm (just above body Ø85 mm)
        "grippable_sweet_low_m":       0.018,   # 18 mm — start of plateau
        "grippable_sweet_high_m":      0.090,   # 90 mm — end of plateau
        "encompassing_diameter_max_m": 0.090,   # body encompass in WIDE
        "finger_spread_candidates_m":  (0.025, 0.050, 0.075, 0.100),
        "seed_count":                  28,
        "patch_radius_m":              0.018,
    },

    # ── Rolă de bandă ───────────────────────────────────────────────
    # Outer Ø100 mm → WIDE mode.  Height 50 mm → small spreads only.
    # Two natural grip zones:
    #   • across height (≈50 mm)    — BASIC or WIDE fingertip
    #   • around outer rim (Ø100 mm) — WIDE encompassing
    "roll_of_tape": {
        "display_name": "Rolă de bandă",
        "grippable_diameter_min_m":    0.030,   # 30 mm
        "grippable_diameter_max_m":    0.110,   # 110 mm (outer Ø100 mm + margin)
        "grippable_sweet_low_m":       0.040,   # 40 mm (height grip ≈50 mm ±10)
        "grippable_sweet_high_m":      0.105,   # 105 mm (outer rim grip)
        "encompassing_diameter_max_m": 0.110,   # outer rim encompass in WIDE
        "finger_spread_candidates_m":  (0.025, 0.040, 0.050),  # object only 50 mm tall
        "seed_count":                  20,
        "patch_radius_m":              0.016,
    },

    # ── Șurubelniță ─────────────────────────────────────────────────
    # Handle Ø20 mm → BASIC / PINCH sweet-spot.
    # Shaft  Ø5 mm  → SCISSOR mode (too thin for other modes).
    # Very elongated (205 mm) → approach preferred across the handle.
    "screwdriver": {
        "display_name": "Șurubelniță",
        "grippable_diameter_min_m":    0.004,   # 4 mm  (shaft floor)
        "grippable_diameter_max_m":    0.025,   # 25 mm (just above handle Ø20 mm)
        "grippable_sweet_low_m":       0.016,   # 16 mm — start of handle plateau
        "grippable_sweet_high_m":      0.022,   # 22 mm — end of handle plateau
        "encompassing_diameter_max_m": 0.025,   # handle fully encompassed
        "finger_spread_candidates_m":  (0.025, 0.050, 0.075),  # spread along length
        "seed_count":                  16,
        "patch_radius_m":              0.012,   # smaller patch for thin object
    },

    # ── Wrench ──────────────────────────────────────────────────────
    # Handle Ø20-25 mm → BASIC / PINCH. Head wider but not graspable.
    # Elongated (~200 mm) → gripper aligns with handle length.
    "wrench": {
        "display_name": "Wrench",
        "grippable_diameter_min_m":    0.012,
        "grippable_diameter_max_m":    0.030,
        "grippable_sweet_low_m":       0.018,
        "grippable_sweet_high_m":      0.026,
        "encompassing_diameter_max_m": 0.030,
        "finger_spread_candidates_m":  (0.025, 0.050, 0.075),
        "seed_count":                  16,
        "patch_radius_m":              0.012,
    },

    # ── Mouse ────────────────────────────────────────────────────────
    # Height 30 mm → BASIC encompassing from the side.
    # Width  60 mm → BASIC fingertip.
    # Length 100 mm → moderate elongation (gripper aligns with length).
    "computer_mouse": {
        "display_name": "Mouse",
        "grippable_diameter_min_m":    0.022,   # 22 mm
        "grippable_diameter_max_m":    0.065,   # 65 mm (width 60 mm + margin)
        "grippable_sweet_low_m":       0.025,   # 25 mm (height grip ≈30 mm)
        "grippable_sweet_high_m":      0.060,   # 60 mm (width grip)
        "encompassing_diameter_max_m": 0.065,   # fits WIDE encompassing
        "finger_spread_candidates_m":  (0.025, 0.050, 0.075),
        "seed_count":                  24,
        "patch_radius_m":              0.015,
    },

    # ── Lotion bottle ────────────────────────────────────────────────
    # bbox ≈ 76×218×121 mm, body Ø ≈ 73 mm, flexible=True.
    # Diameter scoring is intentionally kept at generic limits (70 mm max)
    # so that the rigid baseline gets thick_score=0 on the body, while the
    # flexible path (15% compression → effective Ø ≈ 62 mm) unlocks it.
    # This maximises the visible difference across the 3-way thesis comparison.
    # Only patch sampling / search is tuned for this object's size.
    "lotion_bottle": {
        "display_name": "Lotion Bottle",
        # diameter limits intentionally not set → fall back to generic 20-70 mm
        "encompassing_diameter_max_m": 0.060,   # generic encompass threshold
        "finger_spread_candidates_m":  (0.025, 0.050, 0.075, 0.100),  # tall object
        "seed_count":                  32,
        "patch_radius_m":              0.022,   # slightly larger for a bottle
    },
}


# ── Per-object parameter override ──────────────────────────────────
# Applied automatically at import time when ACTIVE_OBJECT is set.
# run_grasps_on_sam3_outputs.py writes ACTIVE_OBJECT into this file
# before each scoring run, so every object gets its own tuned params.
#
# Lookup: exact key match first; then substring match to handle aliases
# such as "phillips_screwdriver" → "screwdriver".

if ACTIVE_OBJECT is not None:
    _active_key: str | None = None
    if ACTIVE_OBJECT in OBJECT_CATALOG:
        _active_key = ACTIVE_OBJECT
    else:
        for _k in OBJECT_CATALOG:
            if _k in ACTIVE_OBJECT or ACTIVE_OBJECT in _k:
                _active_key = _k
                break

    if _active_key is not None:
        _p = OBJECT_CATALOG[_active_key]
        GRIPPABLE_DIAMETER_MIN_M    = _p.get("grippable_diameter_min_m",    GRIPPABLE_DIAMETER_MIN_M)
        GRIPPABLE_DIAMETER_MAX_M    = _p.get("grippable_diameter_max_m",    GRIPPABLE_DIAMETER_MAX_M)
        GRIPPABLE_SWEET_LOW_M       = _p.get("grippable_sweet_low_m",       GRIPPABLE_SWEET_LOW_M)
        GRIPPABLE_SWEET_HIGH_M      = _p.get("grippable_sweet_high_m",      GRIPPABLE_SWEET_HIGH_M)
        ENCOMPASSING_DIAMETER_MAX_M = _p.get("encompassing_diameter_max_m", ENCOMPASSING_DIAMETER_MAX_M)
        FINGER_SPREAD_CANDIDATES_M  = _p.get("finger_spread_candidates_m",  FINGER_SPREAD_CANDIDATES_M)
        SEED_COUNT                  = _p.get("seed_count",                  SEED_COUNT)
        PATCH_RADIUS_M              = _p.get("patch_radius_m",              PATCH_RADIUS_M)
        del _p
    del _active_key
