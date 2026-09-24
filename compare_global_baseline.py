#!/usr/bin/env python3
"""
Compare the current multi-patch analytical grasp pipeline against a simpler
global-PCA baseline on the same object and in geometry-only mode.

The baseline keeps the same contact generator and the same scorer, but uses
just ONE global frame built on the whole object instead of many local patches.
This is a fair and simple sanity check:
  - if the full pipeline wins, patch search is useful
  - if both are identical, the current object may be too easy
  - if the baseline wins, the local search/constraints need inspection
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

# Force geometry-only scoring for the baseline.
os.environ["USE_HEATMAP"] = "0"

from compute_frames import compute_frame
from generate_contacts import generate_contacts
from grasp_config import (
    FRAME_DIR,
    FRAME_PREFIX,
    INTRINSICS_PATH_NAME,
    PATCH_MIN_POINTS,
)
from grasp_helpers import read_open3d_binary_ply, save_json
from grasp_types import Patch
from object_dimensions import get_object_dimensions
from score_grasps import score_and_rank


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULT_PATHS = (
    ROOT / "grasp_scores_no_heatmap.json",
    ROOT / "grasp_scores.json",
)

NPZ_PATH = FRAME_DIR / f"{FRAME_PREFIX}_grasp_patch_data.npz"
PATCH_PLY_PATH = FRAME_DIR / f"{FRAME_PREFIX}_grasp_patch_cloud.ply"
OBJECT_PLY_PATH = FRAME_DIR / f"{FRAME_PREFIX}_object_cloud.ply"
SCENE_PLY_PATH = FRAME_DIR / f"{FRAME_PREFIX}_scene_cloud.ply"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec * 0.0
    return vec / norm


def _load_intrinsics() -> tuple[float, float, float, float]:
    data = json.loads((Path(__file__).resolve().parent / INTRINSICS_PATH_NAME).read_text())
    return data["fx"], data["fy"], data["cx"], data["cy"]


def _project_3d_to_pixel(points: np.ndarray, fx, fy, cx, cy) -> np.ndarray:
    z = np.where(np.abs(points[:, 2]) > 1e-6, points[:, 2], 1e-6)
    u = fx * points[:, 0] / z + cx
    v = fy * points[:, 1] / z + cy
    return np.column_stack((u, v)).astype(np.int32)


def _filter_cup_exterior_search_space(
    points: np.ndarray,
    pixels: np.ndarray,
    normals: np.ndarray,
    object_center: np.ndarray,
    object_principal_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0:
        return points, pixels, normals

    centered = points - object_center
    axis = _normalize(np.asarray(object_principal_axis, dtype=np.float64))
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
        idx = np.flatnonzero((axial >= lo) & (axial <= hi if i == bin_count - 1 else axial < hi))
        if len(idx) < 8:
            continue

        local_max_radius = float(np.max(radial_dist[idx]))
        local_keep = radial_dist[idx] >= (radius_ratio * local_max_radius)
        local_keep &= outward_alignment[idx] >= min_outward_alignment
        keep_mask[idx[local_keep]] = True

    if int(np.sum(keep_mask)) < min_keep_points:
        global_keep = radial_dist >= float(np.quantile(radial_dist, 0.70))
        global_keep &= outward_alignment >= min_outward_alignment
        if int(np.sum(global_keep)) >= min_keep_points:
            keep_mask = global_keep

    kept = int(np.sum(keep_mask))
    if kept < max(PATCH_MIN_POINTS, 64):
        return points, pixels, normals
    return points[keep_mask], pixels[keep_mask], normals[keep_mask]


def _object_geometry(object_points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
    object_center = object_points.mean(axis=0)
    bbox_extent = object_points.max(axis=0) - object_points.min(axis=0)
    object_scale_m = float(np.linalg.norm(bbox_extent))

    centered = object_points - object_center
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    principal_axis = _normalize(eigvecs[:, 0])

    lam1 = float(max(eigvals[0], 1e-12))
    lam2 = float(max(eigvals[1], 0.0))
    elongation = float(np.clip(1.0 - (lam2 / lam1), 0.0, 1.0))
    return object_center, object_scale_m, principal_axis, elongation


def _candidate_points(candidate: dict[str, Any]) -> list[np.ndarray]:
    return [
        np.array(contact["point_xyz_m"], dtype=np.float64)
        for contact in candidate["contacts"]
    ]


def _triangle_centroid(pts: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.array(pts, dtype=np.float64), axis=0)


def _approach_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    au = _normalize(a)
    bu = _normalize(b)
    dot = float(np.clip(au @ bu, -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _contact_rms_m(a_pts: list[np.ndarray], b_pts: list[np.ndarray]) -> float:
    perms = [(0, 1, 2), (0, 2, 1)]
    best = float("inf")
    for perm in perms:
        sq = 0.0
        for i, j in enumerate(perm):
            diff = a_pts[i] - b_pts[j]
            sq += float(diff @ diff)
        best = min(best, math.sqrt(sq / 3.0))
    return best


def _resolve_result_path(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    for candidate in DEFAULT_RESULT_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No result JSON found. Expected grasp_scores_no_heatmap.json or grasp_scores.json.")


def _build_global_patch(points: np.ndarray) -> Patch:
    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    seed_index = int(np.argmin(dists))
    return Patch(
        patch_id=0,
        seed_index=seed_index,
        seed_point=points[seed_index],
        point_indices=np.arange(len(points), dtype=np.int32),
        mean_heatmap=0.0,
        visible_edge_score=0.0,
    )


def _best_candidate_dict(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid_candidates = [candidate for candidate in candidates if candidate["scores"]["valid"]]
    if valid_candidates:
        return max(valid_candidates, key=lambda item: item["scores"]["final_score"])
    return max(candidates, key=lambda item: item["scores"]["final_score"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare current no-heatmap pipeline result against a simple global-PCA baseline."
    )
    parser.add_argument(
        "--result",
        help="Path to current no-heatmap result JSON. Defaults to grasp_scores_no_heatmap.json, then grasp_scores.json.",
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON path. Defaults to grasp_global_baseline_comparison.json in project root.",
    )
    args = parser.parse_args()

    result_path = _resolve_result_path(args.result)
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else ROOT / "grasp_global_baseline_comparison.json"
    )

    result_data = _load_json(result_path)

    npz = np.load(NPZ_PATH, allow_pickle=True)
    camera_origin = np.asarray(npz["camera_origin"], dtype=np.float64)
    object_points, object_normals = read_open3d_binary_ply(OBJECT_PLY_PATH)
    scene_points, _ = read_open3d_binary_ply(SCENE_PLY_PATH) if SCENE_PLY_PATH.exists() else (object_points, None)
    patch_points_ply, _ = read_open3d_binary_ply(PATCH_PLY_PATH)

    fx, fy, cx, cy = _load_intrinsics()
    object_pixels = _project_3d_to_pixel(object_points, fx, fy, cx, cy)

    object_center, object_scale_m, object_principal_axis, object_elongation = _object_geometry(object_points)
    object_prior = get_object_dimensions(FRAME_PREFIX)

    grasp_points = object_points
    grasp_pixels = object_pixels
    grasp_normals = object_normals
    if object_prior is not None and object_prior.canonical_name == "cup":
        grasp_points, grasp_pixels, grasp_normals = _filter_cup_exterior_search_space(
            grasp_points,
            grasp_pixels,
            grasp_normals,
            object_center,
            object_principal_axis,
        )

    global_patch = _build_global_patch(grasp_points)
    global_frame = compute_frame(global_patch, grasp_points, camera_origin)
    baseline_candidates = generate_contacts(
        [global_frame],
        grasp_points,
        grasp_pixels,
        grasp_normals,
        camera_origin,
        heatmap=None,
        object_principal_axis=object_principal_axis,
        point_heat_values=None,
    )
    baseline_scored = score_and_rank(
        baseline_candidates,
        object_center,
        object_scale_m,
        object_points,
        scene_points,
        object_principal_axis,
        object_elongation,
        object_prior,
    )
    if not baseline_scored:
        raise RuntimeError("Global baseline produced no candidates.")

    baseline_dicts = []
    from run_grasps import _scored_to_dict  # imported lazily after forcing geometry-only env

    baseline_dicts = [_scored_to_dict(item) for item in baseline_scored]
    baseline_best = _best_candidate_dict(baseline_dicts)

    pipeline_candidates = result_data.get("all_candidates", [])
    if not pipeline_candidates:
        raise ValueError(f"No candidates found in {result_path}")
    pipeline_best = _best_candidate_dict(pipeline_candidates)

    baseline_pts = _candidate_points(baseline_best)
    pipeline_pts = _candidate_points(pipeline_best)
    centroid_distance_m = float(np.linalg.norm(
        _triangle_centroid(baseline_pts) - _triangle_centroid(pipeline_pts)
    ))
    approach_angle_deg = _approach_angle_deg(
        np.array(baseline_best["approach_direction_xyz_unit"], dtype=np.float64),
        np.array(pipeline_best["approach_direction_xyz_unit"], dtype=np.float64),
    )
    contact_rms_m = _contact_rms_m(baseline_pts, pipeline_pts)

    report = {
        "frame_prefix": FRAME_PREFIX,
        "result_path": str(result_path),
        "baseline_type": "global_pca_single_patch",
        "pipeline_best": pipeline_best,
        "baseline_best": baseline_best,
        "summary": {
            "pipeline_candidate_count": len(pipeline_candidates),
            "baseline_candidate_count": len(baseline_dicts),
            "pipeline_valid": bool(pipeline_best["scores"]["valid"]),
            "baseline_valid": bool(baseline_best["scores"]["valid"]),
            "pipeline_score": pipeline_best["scores"]["final_score"],
            "baseline_score": baseline_best["scores"]["final_score"],
            "pipeline_mode": pipeline_best["gripper_mode"],
            "baseline_mode": baseline_best["gripper_mode"],
            "pipeline_patch_id": pipeline_best["patch_id"],
            "baseline_patch_id": baseline_best["patch_id"],
            "contact_rms_m": round(contact_rms_m, 6),
            "contact_rms_cm": round(contact_rms_m * 100.0, 3),
            "centroid_distance_m": round(centroid_distance_m, 6),
            "centroid_distance_cm": round(centroid_distance_m * 100.0, 3),
            "approach_angle_deg": round(approach_angle_deg, 3),
            "same_mode": bool(pipeline_best["gripper_mode"] == baseline_best["gripper_mode"]),
            "pipeline_beats_baseline": bool(
                pipeline_best["scores"]["valid"]
                and (
                    (not baseline_best["scores"]["valid"])
                    or pipeline_best["scores"]["final_score"] >= baseline_best["scores"]["final_score"]
                )
            ),
        },
    }

    save_json(output_path, report)

    summary = report["summary"]
    print("\nGlobal baseline comparison")
    print(f"  Result:   {result_path}")
    print(f"  Output:   {output_path}")
    print(f"  Pipeline: patch={summary['pipeline_patch_id']} mode={summary['pipeline_mode']} "
          f"score={summary['pipeline_score']:.4f} valid={summary['pipeline_valid']}")
    print(f"  Baseline: patch={summary['baseline_patch_id']} mode={summary['baseline_mode']} "
          f"score={summary['baseline_score']:.4f} valid={summary['baseline_valid']}")
    print(f"  Contact RMS:     {summary['contact_rms_cm']:.2f} cm")
    print(f"  Centroid delta:  {summary['centroid_distance_cm']:.2f} cm")
    print(f"  Approach delta:  {summary['approach_angle_deg']:.1f} deg")
    print(f"  Same mode:       {summary['same_mode']}")
    print(f"  Pipeline better: {summary['pipeline_beats_baseline']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
