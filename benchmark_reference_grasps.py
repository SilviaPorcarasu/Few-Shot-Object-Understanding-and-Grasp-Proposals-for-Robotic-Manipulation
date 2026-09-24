#!/usr/bin/env python3
"""
Compare local grasp results against external reference grasps.

Typical use:
  1. Run the analytical baseline without heatmap and save
     `grasp_scores_no_heatmap.json`.
  2. Export one or more reference grasps (from GraspIt!, a paper repo,
     or manual annotation) into a compact JSON file.
  3. Run this script to measure how closely the local candidates match
     the references.

The script does NOT assume a learning model. It compares:
  - contact triangle geometry
  - grasp centroid
  - approach direction
  - optional mode label agreement

Reference JSON schema:
{
  "frame_prefix": "blue_cup",
  "references": [
    {
      "label": "graspit_top1",
      "gripper_mode": "wide",
      "approach_direction_xyz_unit": [0.0, 0.0, 1.0],
      "contacts": [
        {"name": "contact_1", "point_xyz_m": [x, y, z]},
        {"name": "contact_2", "point_xyz_m": [x, y, z]},
        {"name": "contact_3", "point_xyz_m": [x, y, z]}
      ]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULT_PATHS = (
    ROOT / "grasp_scores_no_heatmap.json",
    ROOT / "grasp_scores.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec * 0.0
    return vec / norm


def _contact_points_from_candidate(candidate: dict[str, Any]) -> list[np.ndarray]:
    return [
        np.array(contact["point_xyz_m"], dtype=np.float64)
        for contact in candidate["contacts"]
    ]


def _contact_points_from_reference(reference: dict[str, Any]) -> list[np.ndarray]:
    return [
        np.array(contact["point_xyz_m"], dtype=np.float64)
        for contact in reference["contacts"]
    ]


def _triangle_side_lengths(pts: list[np.ndarray]) -> np.ndarray:
    p1, p2, p3 = pts
    return np.array([
        float(np.linalg.norm(p1 - p2)),
        float(np.linalg.norm(p1 - p3)),
        float(np.linalg.norm(p2 - p3)),
    ])


def _triangle_centroid(pts: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.array(pts, dtype=np.float64), axis=0)


def _approach_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    au = _normalize(a)
    bu = _normalize(b)
    dot = float(np.clip(au @ bu, -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def _contact_rms_m(ref_pts: list[np.ndarray], cand_pts: list[np.ndarray]) -> float:
    """
    F1 is semantic thumb contact and should stay fixed.
    F2/F3 are interchangeable, so we test both assignments.
    """
    if len(ref_pts) != 3 or len(cand_pts) != 3:
        return float("inf")

    perms = [
        (0, 1, 2),
        (0, 2, 1),
    ]
    best = float("inf")
    for perm in perms:
        sq = 0.0
        for ref_idx, cand_idx in enumerate(perm):
            diff = ref_pts[ref_idx] - cand_pts[cand_idx]
            sq += float(diff @ diff)
        rms = math.sqrt(sq / 3.0)
        best = min(best, rms)
    return best


def _triangle_side_rms_m(ref_pts: list[np.ndarray], cand_pts: list[np.ndarray]) -> float:
    ref_sides = np.sort(_triangle_side_lengths(ref_pts))
    cand_sides = np.sort(_triangle_side_lengths(cand_pts))
    diff = ref_sides - cand_sides
    return math.sqrt(float(np.mean(diff * diff)))


def _compare_reference_to_candidate(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    ref_pts = _contact_points_from_reference(reference)
    cand_pts = _contact_points_from_candidate(candidate)

    ref_centroid = _triangle_centroid(ref_pts)
    cand_centroid = _triangle_centroid(cand_pts)
    centroid_distance_m = float(np.linalg.norm(ref_centroid - cand_centroid))

    ref_approach = np.array(reference["approach_direction_xyz_unit"], dtype=np.float64)
    cand_approach = np.array(candidate["approach_direction_xyz_unit"], dtype=np.float64)
    approach_angle_deg = _approach_angle_deg(ref_approach, cand_approach)

    contact_rms_m = _contact_rms_m(ref_pts, cand_pts)
    side_rms_m = _triangle_side_rms_m(ref_pts, cand_pts)
    mode_match = reference.get("gripper_mode") == candidate.get("gripper_mode")

    # A compact agreement score in [0, 1], useful for ranking candidates
    # against one external reference. Lower geometric errors and lower
    # angular error increase the score.
    contact_term = math.exp(-contact_rms_m / 0.02)          # 2 cm scale
    centroid_term = math.exp(-centroid_distance_m / 0.02)   # 2 cm scale
    side_term = math.exp(-side_rms_m / 0.01)                # 1 cm scale
    angle_term = math.exp(-approach_angle_deg / 25.0)       # 25 deg scale
    mode_term = 1.0 if mode_match else 0.0
    agreement_score = (
        0.35 * contact_term
        + 0.20 * centroid_term
        + 0.20 * side_term
        + 0.15 * angle_term
        + 0.10 * mode_term
    )

    return {
        "candidate_patch_id": candidate.get("patch_id"),
        "candidate_mode": candidate.get("gripper_mode"),
        "candidate_spread_mm": candidate.get("finger_spread_mm"),
        "candidate_rotation_deg": candidate.get("rotation_deg"),
        "candidate_valid": candidate.get("scores", {}).get("valid"),
        "candidate_final_score": candidate.get("scores", {}).get("final_score"),
        "contact_rms_m": round(contact_rms_m, 6),
        "contact_rms_cm": round(contact_rms_m * 100.0, 3),
        "centroid_distance_m": round(centroid_distance_m, 6),
        "centroid_distance_cm": round(centroid_distance_m * 100.0, 3),
        "triangle_side_rms_m": round(side_rms_m, 6),
        "triangle_side_rms_cm": round(side_rms_m * 100.0, 3),
        "approach_angle_deg": round(approach_angle_deg, 3),
        "mode_match": mode_match,
        "agreement_score": round(agreement_score, 6),
    }


def _best_match_for_reference(
    reference: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    valid_only: bool,
) -> dict[str, Any]:
    pool = candidates
    if valid_only:
        valid_candidates = [
            candidate for candidate in candidates
            if candidate.get("scores", {}).get("valid", False)
        ]
        if valid_candidates:
            pool = valid_candidates

    comparisons = [
        _compare_reference_to_candidate(reference, candidate)
        for candidate in pool
    ]
    comparisons.sort(key=lambda item: item["agreement_score"], reverse=True)
    best = comparisons[0]
    best["pool_size"] = len(pool)
    return best


def _resolve_result_path(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    for candidate in DEFAULT_RESULT_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No result JSON found. Expected grasp_scores_no_heatmap.json or grasp_scores.json."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare one local grasp result JSON against external reference grasps."
    )
    parser.add_argument(
        "--result",
        help="Path to local grasp result JSON. Defaults to grasp_scores_no_heatmap.json, then grasp_scores.json.",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference grasp JSON exported from GraspIt!/paper/manual annotation.",
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON path. Defaults to grasp_reference_benchmark.json in project root.",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Allow matching against invalid local candidates too. By default only valid ones are used if any exist.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result_path = _resolve_result_path(args.result)
    reference_path = Path(args.reference).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else ROOT / "grasp_reference_benchmark.json"
    )

    result_data = _load_json(result_path)
    reference_data = _load_json(reference_path)

    candidates = result_data.get("all_candidates", [])
    if not candidates:
        raise ValueError(f"No candidates found in {result_path}")

    references = reference_data.get("references", [])
    if not references:
        raise ValueError(f"No references found in {reference_path}")

    comparisons = []
    for reference in references:
        best = _best_match_for_reference(
            reference,
            candidates,
            valid_only=not args.include_invalid,
        )
        comparisons.append({
            "reference_label": reference.get("label", "reference"),
            "reference_mode": reference.get("gripper_mode"),
            "best_match": best,
        })

    summary = {
        "frame_prefix": result_data.get("frame_prefix"),
        "result_path": str(result_path),
        "reference_path": str(reference_path),
        "used_valid_only": not args.include_invalid,
        "reference_count": len(references),
        "candidate_count": len(candidates),
        "comparisons": comparisons,
    }

    output_path.write_text(json.dumps(summary, indent=2))

    print("\nReference benchmark summary")
    print(f"  Result:    {result_path}")
    print(f"  Reference: {reference_path}")
    print(f"  Output:    {output_path}")
    print("")
    for item in comparisons:
        best = item["best_match"]
        print(
            f"- {item['reference_label']}: "
            f"mode={best['candidate_mode']} "
            f"agreement={best['agreement_score']:.3f} "
            f"contact_rms={best['contact_rms_cm']:.2f}cm "
            f"approach_delta={best['approach_angle_deg']:.1f}deg "
            f"valid={best['candidate_valid']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
