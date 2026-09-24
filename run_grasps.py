#!/usr/bin/env python3
"""
Entry point for the grasp generation pipeline.

Reads:
    <FRAME_DIR>/<FRAME_PREFIX>_object_cloud.ply
        - FULL object point cloud with normals (the entire visible
          surface of the banana)
    <FRAME_DIR>/<FRAME_PREFIX>_grasp_patch_data.npz
        - camera_origin and other metadata
    <HEATMAP_DIR>/<frame_name>_pred_gray.png
        - ML affordance heatmap (0..255)
    camera_intrinsics.json
        - fx, fy, cx, cy recovered from the upstream point→pixel mapping

The heatmap is projected onto the FULL object cloud via the camera
intrinsics (perspective projection u = fx*X/Z + cx, v = fy*Y/Z + cy),
giving every 3D object point its own ML affordance value. Points below
HEATMAP_FILTER_THRESHOLD are discarded; the rest are the search space
for grasp candidates.

Pipeline (4 stages):
    1. select_patches    — split filtered cloud into local sub-zones
    2. compute_frames    — build a PCA frame on each patch
    3. generate_contacts — place 3 contacts at 120° + lookup heatmap
    4. score_and_rank    — evaluate and sort

Output:
    grasp_scores.json    — all grasps with scores, sorted best-first
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from compute_frames import compute_frames
from generate_contacts import generate_contacts
from grasp_config import (
    FRAME_DIR,
    FRAME_PREFIX,
    GRASP_SEARCH_STRATEGY,
    HEATMAP_DIR,
    HEATMAP_FILTER_THRESHOLD,
    HEATMAP_LOCAL_PATCH_MIN,
    HEATMAP_LOCAL_PATCH_TOPK,
    INTRINSICS_PATH_NAME,
    PATCH_MIN_POINTS,
    RGB_H,
    RGB_W,
    ROBOTIQ_3F_MAX_OPENING_M,
    USE_HEATMAP,
)
from grasp_helpers import (
    project_sparse_point_heat_to_object,
    read_open3d_binary_ply,
    save_json,
)
from grasp_types import ScoredGrasp
from object_dimensions import ObjectDimensions, get_object_dimensions
from score_grasps import score_and_rank
from select_patches import build_global_patch, select_patches


# ── File paths ──────────────────────────────────────────────────────

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
NPZ_PATH = FRAME_DIR / f"{FRAME_PREFIX}_grasp_patch_data.npz"
PATCH_PLY_PATH = FRAME_DIR / f"{FRAME_PREFIX}_grasp_patch_cloud.ply"
OBJECT_PLY_PATH = FRAME_DIR / f"{FRAME_PREFIX}_object_cloud.ply"
OBJECT_PLY_OVERRIDE = Path(
    os.environ["GRASP_OBJECT_PLY_OVERRIDE"]
) if "GRASP_OBJECT_PLY_OVERRIDE" in os.environ else None
SCENE_PLY_PATH = FRAME_DIR / f"{FRAME_PREFIX}_scene_cloud.ply"
OUTPUT_PATH = PROJECT_ROOT / "grasp_scores.json"


# ── Loading ─────────────────────────────────────────────────────────

def _load_heatmap(frame_name: str) -> np.ndarray | None:
    """Load the ML affordance heatmap for this frame, if available."""
    candidate = HEATMAP_DIR / f"{frame_name}_pred_gray.png"
    if not candidate.exists():
        print(f"  Warning: heatmap not found at {candidate}")
        return None
    try:
        from PIL import Image
        img = np.asarray(Image.open(candidate).convert("L"))
    except ImportError:
        import imageio.v3 as iio
        img = np.asarray(iio.imread(candidate))
        if img.ndim == 3:
            img = img[..., 0]
    print(f"  Heatmap: {img.shape} from {candidate.name}")
    return img


def _scale_pixels_to_heatmap(
    pixels: np.ndarray, heatmap: np.ndarray | None,
    rgb_size: tuple[int, int] = (1280, 720),
) -> np.ndarray:
    """
    Scale pixel coords from the original RGB resolution to the heatmap.

    The npz stores pixels in the original RGB image space (1280x720 by default).
    The heatmap was generated at a smaller resolution (e.g. 640x384), so we
    rescale to match.
    """
    if heatmap is None:
        return pixels
    rgb_w, rgb_h = rgb_size
    hm_h, hm_w = heatmap.shape[:2]
    if rgb_w == hm_w and rgb_h == hm_h:
        return pixels
    sx = hm_w / rgb_w
    sy = hm_h / rgb_h
    scaled = np.empty_like(pixels)
    scaled[:, 0] = np.clip((pixels[:, 0] * sx).astype(np.int32), 0, hm_w - 1)
    scaled[:, 1] = np.clip((pixels[:, 1] * sy).astype(np.int32), 0, hm_h - 1)
    return scaled


# ── JSON export ─────────────────────────────────────────────────────

def _vec(v: np.ndarray) -> list[float]:
    return [float(x) for x in v]


def _vec_cm(v: np.ndarray) -> list[float]:
    return [round(float(x) * 100.0, 4) for x in v]


def _scored_to_dict(sg: ScoredGrasp) -> dict:
    c = sg.candidate
    f = c.frame
    return {
        "patch_id": f.patch.patch_id,
        "gripper_mode": c.gripper_mode,
        "finger_spread_mm": round(c.finger_spread_m * 1000.0, 2),
        "rotation_deg": round(c.rotation_deg, 2),
        "seed_index": f.patch.seed_index,
        "patch_point_count": len(f.patch.point_indices),
        "seed_point_xyz_m": _vec(f.patch.seed_point),
        "seed_point_xyz_cm": _vec_cm(f.patch.seed_point),
        "patch_centroid_xyz_m": _vec(f.centroid),
        "patch_centroid_xyz_cm": _vec_cm(f.centroid),
        "patch_pca_eigenvalues": _vec(f.pca_eigenvalues),
        "tangent_1_xyz_unit": _vec(f.tangent_1),
        "tangent_2_xyz_unit": _vec(f.tangent_2),
        "normal_xyz_unit": _vec(f.normal),
        "approach_direction_xyz_unit": _vec(c.approach_direction),
        "visible_contact_count": c.visible_count,
        "contacts": [
            {
                "name": ct.name,
                "angle_deg": ct.angle_deg,
                "point_xyz_m": _vec(ct.point),
                "point_xyz_cm": _vec_cm(ct.point),
                "normal_xyz_unit": _vec(ct.normal),
                "visible": ct.visible,
                "distance_to_visible_support_m": ct.distance_to_visible_m,
                "distance_to_visible_support_cm": round(ct.distance_to_visible_m * 100.0, 3),
                "pixel_uv": list(ct.pixel) if ct.pixel else None,
                "heatmap_value": ct.heatmap_value,
            }
            for ct in c.contacts
        ],
        "scores": {
            "valid": bool(sg.valid),
            "final_score": sg.final_score,
            "grip_type": sg.grip_type,
            "heatmap_score": sg.heatmap_score,
            "thickness_score": sg.thickness_score,
            "local_diameter_cm": round(sg.local_diameter_m * 100.0, 3),
            "local_diameter_mm": round(sg.local_diameter_m * 1000.0, 2),
            "patch_planarity_score": sg.patch_planarity,
            "triangle_score": sg.triangle_score,
            "triangle_area_m2": sg.triangle_area_m2,
            "alignment_score": sg.alignment_score,
            "feature_score": sg.feature_score,
            "distance_grasp_to_com_m": sg.distance_to_com_m,
            "distance_grasp_to_com_cm": round(sg.distance_to_com_m * 100.0, 3),
            "visible_fraction": sg.visible_fraction,
            "required_opening_m": sg.required_opening_m,
            "required_opening_cm": round(sg.required_opening_m * 100.0, 3),
            "required_opening_mm": round(sg.required_opening_m * 1000.0, 3),
            "opening_feasible": bool(sg.opening_feasible),
            "contacts_reachable": bool(sg.contacts_reachable),
            "palm_collision_free": bool(sg.palm_collision_free),
            "fingers_collision_free": bool(sg.fingers_collision_free),
            "approach_clear": bool(sg.approach_clear),
            "support_clear": bool(sg.support_clear),
            "top_entry_like": bool(sg.top_entry_like),
            "opposition_score": sg.opposition_score,
            "wrench_score": sg.wrench_score,
            "object_flexible": bool(sg.object_flexible),
            "flex_score": sg.flex_score,
        },
    }


def export_results(
    scored: list[ScoredGrasp],
    object_center: np.ndarray,
    object_scale_m: float,
    camera_origin: np.ndarray,
    measured_object_scale_m: float,
    object_prior: ObjectDimensions | None,
    object_flexible: bool = False,
) -> None:
    all_dicts = [_scored_to_dict(sg) for sg in scored]
    best = next((d for d in all_dicts if d["scores"]["valid"]), None)
    if best is None and all_dicts:
        best = all_dicts[0]

    report = {
        "frame_prefix": FRAME_PREFIX,
        "object_flexible": bool(object_flexible),
        "input_npz": str(NPZ_PATH),
        "input_object_cloud": str(OBJECT_PLY_PATH),
        "camera_origin_xyz_m": _vec(camera_origin),
        "camera_origin_xyz_cm": _vec_cm(camera_origin),
        "object_center_xyz_m": _vec(object_center),
        "object_center_xyz_cm": _vec_cm(object_center),
        "measured_object_scale_m": float(measured_object_scale_m),
        "measured_object_scale_cm": round(float(measured_object_scale_m) * 100.0, 3),
        "scoring_object_scale_m": float(object_scale_m),
        "scoring_object_scale_cm": round(float(object_scale_m) * 100.0, 3),
        "object_dimension_prior": (
            object_prior.as_dict() if object_prior is not None else None
        ),
        "robotiq_3f_max_opening_cm": round(ROBOTIQ_3F_MAX_OPENING_M * 100.0, 3),
        "robotiq_3f_max_opening_mm": round(ROBOTIQ_3F_MAX_OPENING_M * 1000.0, 3),
        "best_candidate": best,
        "all_candidates": all_dicts,
    }

    save_json(OUTPUT_PATH, report)
    print(f"Saved {len(all_dicts)} scored grasps to {OUTPUT_PATH.name}")


# ── Main pipeline ───────────────────────────────────────────────────

def _load_intrinsics():
    """Load fx, fy, cx, cy from camera_intrinsics.json next to this file."""
    path = CODE_ROOT / INTRINSICS_PATH_NAME
    data = json.loads(path.read_text())
    return data["fx"], data["fy"], data["cx"], data["cy"]


def _project_3d_to_pixel(points: np.ndarray, fx, fy, cx, cy) -> np.ndarray:
    """Perspective projection: 3D camera-frame points → pixel coordinates."""
    Z = np.where(np.abs(points[:, 2]) > 1e-6, points[:, 2], 1e-6)
    u = fx * points[:, 0] / Z + cx
    v = fy * points[:, 1] / Z + cy
    return np.column_stack((u, v))


def _filter_cup_exterior_search_space(
    points: np.ndarray,
    pixels: np.ndarray,
    normals: np.ndarray,
    heat_values: np.ndarray | None,
    object_center: np.ndarray,
    object_principal_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """
    Restrict cup/mug grasp search to the exterior shell of the object.

    Semantic cup reconstructions can contain both the outer wall and the
    inner cavity. For grasping we currently want the gripper to land on the
    OUTSIDE of the cup, not on interior surfaces. We therefore keep only
    points that stay close to the maximum radius of the object in their
    local axial slice, plus a soft outward-normal sanity check.
    """
    if len(points) == 0:
        return points, pixels, normals, heat_values, {
            "applied": False,
            "reason": "empty-cloud",
        }

    centered = points - object_center
    axis = np.asarray(object_principal_axis, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)

    axial = centered @ axis
    radial = centered - np.outer(axial, axis)
    radial_dist = np.linalg.norm(radial, axis=1)

    valid_radial = radial_dist > 1e-6
    outward_alignment = np.full(len(points), -1.0, dtype=np.float64)
    if len(normals) == len(points) and np.any(valid_radial):
        outward_dirs = np.zeros_like(radial)
        outward_dirs[valid_radial] = radial[valid_radial] / radial_dist[valid_radial, None]
        outward_alignment[valid_radial] = np.einsum(
            "ij,ij->i", normals[valid_radial], outward_dirs[valid_radial],
        )

    radius_ratio = 0.85
    min_outward_alignment = -0.10
    min_keep_points = max(PATCH_MIN_POINTS * 2, int(0.10 * len(points)))
    bin_count = min(48, max(12, int(np.sqrt(len(points)) // 2)))
    edges = np.linspace(float(axial.min()), float(axial.max()), bin_count + 1)

    keep_mask = np.zeros(len(points), dtype=bool)
    for i in range(bin_count):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bin_count - 1:
            idx = np.flatnonzero((axial >= lo) & (axial <= hi))
        else:
            idx = np.flatnonzero((axial >= lo) & (axial < hi))
        if len(idx) < 8:
            continue

        local_max_radius = float(np.max(radial_dist[idx]))
        local_keep = radial_dist[idx] >= (radius_ratio * local_max_radius)
        local_keep &= outward_alignment[idx] >= min_outward_alignment
        keep_mask[idx[local_keep]] = True

    # Fallback for sparse / partial cup reconstructions: still bias strongly
    # toward the outer wall using a global radial percentile.
    fallback_used = False
    if int(np.sum(keep_mask)) < min_keep_points:
        global_keep = radial_dist >= float(np.quantile(radial_dist, 0.70))
        global_keep &= outward_alignment >= min_outward_alignment
        if int(np.sum(global_keep)) >= min_keep_points:
            keep_mask = global_keep
            fallback_used = True

    kept = int(np.sum(keep_mask))
    if kept < max(PATCH_MIN_POINTS, 64):
        return points, pixels, normals, heat_values, {
            "applied": False,
            "reason": "too-few-points-after-filter",
            "candidate_points": kept,
            "original_points": int(len(points)),
        }

    filtered_heat = heat_values[keep_mask] if heat_values is not None else None
    return (
        points[keep_mask],
        pixels[keep_mask],
        normals[keep_mask],
        filtered_heat,
        {
            "applied": True,
            "kept_points": kept,
            "original_points": int(len(points)),
            "kept_fraction": float(kept / max(len(points), 1)),
            "fallback_used": fallback_used,
            "radius_ratio": radius_ratio,
            "min_outward_alignment": min_outward_alignment,
        },
    )


def main() -> None:
    print(f"Loading data from {FRAME_DIR.name}/{FRAME_PREFIX} ...")

    # Camera origin from npz (camera frame is anchored at 0,0,0)
    npz = np.load(NPZ_PATH, allow_pickle=True)
    camera_origin = np.asarray(npz["camera_origin"], dtype=np.float64)
    frame_name = str(npz["frame_name"])
    patch_points_from_npz = np.asarray(
        npz["points"], dtype=np.float64,
    ) if "points" in npz.files else np.zeros((0, 3), dtype=np.float64)
    patch_pixels_from_npz = np.asarray(
        npz["pixels"], dtype=np.int32,
    ) if "pixels" in npz.files else np.zeros((len(patch_points_from_npz), 2), dtype=np.int32)
    patch_heat_values = np.asarray(
        npz["point_heat_values"], dtype=np.float64,
    ) if "point_heat_values" in npz.files else None
    object_flexible = bool(np.asarray(npz["object_flexible"]).item()) if "object_flexible" in npz.files else False
    _flex_env = os.environ.get("OBJECT_FLEXIBLE_OVERRIDE", "").strip().lower()
    if _flex_env in ("0", "false", "no"):
        object_flexible = False
    elif _flex_env in ("1", "true", "yes"):
        object_flexible = True

    # Full object cloud — this is now the WHOLE search space, not just patch.
    object_ply_path = OBJECT_PLY_OVERRIDE if OBJECT_PLY_OVERRIDE is not None else OBJECT_PLY_PATH
    object_points, object_normals = read_open3d_binary_ply(object_ply_path)
    if SCENE_PLY_PATH.exists():
        scene_points, _ = read_open3d_binary_ply(SCENE_PLY_PATH)
    else:
        scene_points = object_points
    patch_points_ply, patch_normals = read_open3d_binary_ply(PATCH_PLY_PATH)

    # If the adapted npz already contains 3D heat values, we do not need
    # to go back to a 2D heatmap image just to recover the same signal.
    use_point_heat_values = (
        USE_HEATMAP
        and patch_heat_values is not None
        and len(patch_heat_values) == len(patch_points_ply)
    )

    # Heatmap and camera intrinsics
    heatmap = _load_heatmap(frame_name) if (USE_HEATMAP and not use_point_heat_values) else None
    fx, fy, cx, cy = _load_intrinsics()
    if not USE_HEATMAP:
        print("  Heatmap: DISABLED — geometry-only mode")

    # Project every object point to its pixel in the RGB image, then
    # rescale to the heatmap resolution to look up its affordance value.
    obj_pixels_rgb = _project_3d_to_pixel(object_points, fx, fy, cx, cy)
    if heatmap is not None:
        hm_h, hm_w = heatmap.shape[:2]
        sx, sy = hm_w / RGB_W, hm_h / RGB_H
        obj_pixels = np.empty_like(obj_pixels_rgb, dtype=np.int32)
        obj_pixels[:, 0] = np.clip(
            (obj_pixels_rgb[:, 0] * sx).astype(np.int32), 0, hm_w - 1
        )
        obj_pixels[:, 1] = np.clip(
            (obj_pixels_rgb[:, 1] * sy).astype(np.int32), 0, hm_h - 1
        )
    else:
        obj_pixels = obj_pixels_rgb.astype(np.int32)

    # Object centre + scale
    object_center = object_points.mean(axis=0)
    bbox_extent = object_points.max(axis=0) - object_points.min(axis=0)
    measured_object_scale_m = float(np.linalg.norm(bbox_extent))
    object_prior = get_object_dimensions(FRAME_PREFIX)
    # Real dimensions are used only for final reporting / validation, not
    # during grasp selection. Scoring therefore stays tied to the measured
    # geometry coming from the current reconstruction.
    object_scale_m = measured_object_scale_m

    # Object-level PCA — gives us the principal (longest) axis of the
    # object and how elongated it is. Used by the alignment metric to
    # prefer approaches perpendicular to the long axis (across the
    # banana, across the bottle, ...). For near-isotropic objects
    # (sphere, cube) elongation ≈ 0 and the metric is neutral.
    centred_obj = object_points - object_center
    cov_obj = (centred_obj.T @ centred_obj) / max(len(centred_obj) - 1, 1)
    eigvals_obj, eigvecs_obj = np.linalg.eigh(cov_obj)
    order = np.argsort(eigvals_obj)[::-1]
    eigvals_obj = eigvals_obj[order]
    eigvecs_obj = eigvecs_obj[:, order]
    object_principal_axis = eigvecs_obj[:, 0]
    object_principal_axis = (
        object_principal_axis
        / max(float(np.linalg.norm(object_principal_axis)), 1e-9)
    )
    lam1 = float(max(eigvals_obj[0], 1e-12))
    lam2 = float(max(eigvals_obj[1], 0.0))
    object_elongation = float(np.clip(1.0 - (lam2 / lam1), 0.0, 1.0))

    # There are two valid upstream ways to express "this is a good place
    # to grasp":
    #   1. classic 2D heatmap image that we project onto the full object
    #   2. direct per-point 3D heat values already attached to patch points
    #
    # Symmetric semantic and fused asymmetric outputs naturally fit (2).
    if use_point_heat_values:
        raw_point_heat_values = np.clip(patch_heat_values, 0.0, 1.0)
        grasp_points = object_points
        grasp_pixels = obj_pixels
        grasp_normals = object_normals
        grasp_hm_values = project_sparse_point_heat_to_object(
            object_points, patch_points_ply, raw_point_heat_values,
        )
        heat_source_name = "3D point heat"
    else:
        # Filter the cloud by the heatmap threshold — only the affordance
        # region is searched for grasp candidates.
        if heatmap is not None:
            u_hm = obj_pixels[:, 0]
            v_hm = obj_pixels[:, 1]
            obj_hm_values = heatmap[v_hm, u_hm].astype(np.float64) / 255.0
            keep_mask = obj_hm_values >= HEATMAP_FILTER_THRESHOLD
        else:
            keep_mask = np.ones(len(object_points), dtype=bool)

        grasp_points = object_points[keep_mask]
        grasp_pixels = obj_pixels[keep_mask]
        grasp_normals = object_normals[keep_mask]
        grasp_hm_values = (
            obj_hm_values[keep_mask] if heatmap is not None else None
        )
        heat_source_name = "2D projected heatmap" if heatmap is not None else "geometry only"

    if object_prior is not None and object_prior.canonical_name == "cup":
        grasp_points, grasp_pixels, grasp_normals, grasp_hm_values, exterior_stats = (
            _filter_cup_exterior_search_space(
                grasp_points,
                grasp_pixels,
                grasp_normals,
                grasp_hm_values,
                object_center,
                object_principal_axis,
            )
        )
    else:
        exterior_stats = None

    cam_cm = [round(v * 100.0, 2) for v in camera_origin.tolist()]
    print(f"  Camera origin: {cam_cm} cm")
    print(f"  Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, "
          f"cx={cx:.1f}, cy={cy:.1f}")
    if OBJECT_PLY_OVERRIDE is not None:
        print(f"  Object cloud override: {object_ply_path.name}")
    print(f"  Object: {len(object_points)} points, measured scale "
          f"{measured_object_scale_m*100:.2f} cm")
    print(f"  Object flexible: {object_flexible}")
    if object_prior is not None:
        dims_cm = [round(v * 100.0, 2) for v in object_prior.bounding_box_xyz_m]
        print(f"  Real dimensions reference: {object_prior.canonical_name} "
              f"{dims_cm} cm, measured scoring scale {object_scale_m*100:.2f} cm")
    else:
        print(f"  No real-size prior found for '{FRAME_PREFIX}', "
              f"scoring scale {object_scale_m*100:.2f} cm")
    if use_point_heat_values:
        print(f"  Search source: {heat_source_name} "
              f"({len(grasp_points)} object points)")
        print(f"  Raw 3D heat range: "
              f"{raw_point_heat_values.min():.4f} .. {raw_point_heat_values.max():.4f}")
        print(f"  Projected full-object heat range: "
              f"{grasp_hm_values.min():.4f} .. {grasp_hm_values.max():.4f}")
    elif USE_HEATMAP:
        print(f"  Search source: {heat_source_name}")
        print(f"  Heatmap-filtered (>= {HEATMAP_FILTER_THRESHOLD}): "
              f"{len(grasp_points)}/{len(object_points)} points")
    else:
        print(f"  Search source: {heat_source_name}")
        print(f"  Search space: {len(grasp_points)} points (full object)")
    if exterior_stats is not None:
        if exterior_stats["applied"]:
            kept_pct = 100.0 * exterior_stats["kept_fraction"]
            fallback_note = " (global fallback)" if exterior_stats["fallback_used"] else ""
            print(
                "  Cup exterior shell filter:"
                f" {exterior_stats['kept_points']}/{exterior_stats['original_points']}"
                f" points kept ({kept_pct:.1f}%){fallback_note}"
            )
        else:
            print(
                "  Cup exterior shell filter skipped:"
                f" {exterior_stats['reason']}"
            )
    print()

    # STAGE 1: choose the analytical search strategy.
    if GRASP_SEARCH_STRATEGY == "global_pca_baseline":
        patches = [
            build_global_patch(
                grasp_points,
                grasp_hm_values,
                grasp_pixels=grasp_pixels,
            )
        ]
        if USE_HEATMAP and grasp_hm_values is not None:
            hot_local_patches = [
                p for p in select_patches(
                    grasp_points,
                    grasp_hm_values,
                    grasp_pixels=grasp_pixels,
                )
                if p.mean_heatmap >= HEATMAP_LOCAL_PATCH_MIN
            ][:HEATMAP_LOCAL_PATCH_TOPK]
            for i, patch in enumerate(hot_local_patches, start=1):
                patch.patch_id = i
            patches.extend(hot_local_patches)
        print(
            "  Search strategy: global_pca_baseline "
            f"({len(grasp_points)} points in one global patch)"
        )
        if len(patches) > 1:
            print(
                f"  Heatmap refinement: +{len(patches) - 1} local hot patches "
                f"(top-k={HEATMAP_LOCAL_PATCH_TOPK}, min={HEATMAP_LOCAL_PATCH_MIN:.2f})"
            )
    elif GRASP_SEARCH_STRATEGY == "local_patches":
        patches = select_patches(
            grasp_points,
            grasp_hm_values,
            grasp_pixels=grasp_pixels,
        )
        if grasp_hm_values is not None and patches:
            bright_patches = sum(1 for p in patches if p.mean_heatmap > 0.10)
            print(f"  Search strategy: local_patches")
            print(f"  {len(patches)} patches; "
                  f"top patch mean_heatmap={patches[0].mean_heatmap:.3f}, "
                  f"edge_prior={patches[0].visible_edge_score:.3f}, "
                  f"{bright_patches} above 0.10")
        else:
            print(f"  Search strategy: local_patches")
            print(f"  {len(patches)} patches (geometry-only, no heatmap)")
    else:
        raise ValueError(
            f"Unknown GRASP_SEARCH_STRATEGY='{GRASP_SEARCH_STRATEGY}'. "
            "Use 'global_pca_baseline' or 'local_patches'."
        )

    # STAGE 2: PCA frame on each patch (oriented toward camera)
    frames = compute_frames(patches, grasp_points, camera_origin)
    print(f"  {len(frames)} local frames")

    # STAGE 3: 1-thumb + 2-fingers opposed contacts + heatmap lookup
    candidates = generate_contacts(
        frames, grasp_points, grasp_pixels, grasp_normals,
        camera_origin, heatmap, object_principal_axis,
        point_heat_values=grasp_hm_values if use_point_heat_values else None,
        object_flexible=object_flexible,
    )
    print(f"  {len(candidates)} contact triplets")

    # STAGE 4: Score and rank (full object cloud for collision checks)
    print(f"  Object elongation: {object_elongation:.3f} "
          f"(0=isotropic, 1=line-like)")
    scored = score_and_rank(
        candidates, object_center, object_scale_m, object_points, scene_points,
        object_principal_axis, object_elongation, object_prior,
        object_flexible,
    )
    print(f"  {len(scored)} scored grasps")

    export_results(
        scored, object_center, object_scale_m, camera_origin,
        measured_object_scale_m, object_prior, object_flexible,
    )

    if not scored:
        print("\nNo grasp candidates survived scoring for this object.")
        return

    best = next((g for g in scored if g.valid), scored[0])
    print(f"\nBest grasp: patch {best.candidate.frame.patch.patch_id}, "
          f"mode={best.candidate.gripper_mode}, rot={best.candidate.rotation_deg:.1f}deg, "
          f"grip_type={best.grip_type}, "
          f"score {best.final_score:.4f}, valid={best.valid}")
    print(f"  heatmap={best.heatmap_score:.3f}  "
          f"thickness={best.thickness_score:.3f} (Ø{best.local_diameter_m*100:.2f}cm)  "
          f"triangle={best.triangle_score:.3f}")


if __name__ == "__main__":
    main()
