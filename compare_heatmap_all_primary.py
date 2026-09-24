#!/usr/bin/env python3
"""
Run the clean baseline experiment over all primary reconstructed objects:

  1. full reconstructed object cloud, no heatmap
  2. same full reconstructed object cloud, with heatmap prior

This matches the intended methodology:
  - first validate the analytical 3-finger baseline on the completed object
  - then check whether heatmap patches improve or change the grasp
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from run_grasps_on_sam3_outputs import (
    OUTPUT_ROOT,
    adapt_source,
    build_object_evaluations,
    collect_available_sources,
    update_grasp_config,
)


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
SCORES_PATH = PROJECT_ROOT / "grasp_scores.json"
SUMMARY_PATH = OUTPUT_ROOT / "heatmap_full_reconstruction_summary.json"


def _best_candidate(data: dict[str, Any]) -> dict[str, Any] | None:
    candidates = data.get("all_candidates", [])
    if not candidates:
        return None
    valid = [c for c in candidates if c["scores"]["valid"]]
    ranked = valid if valid else candidates
    return max(ranked, key=lambda c: c["scores"]["final_score"])


def _top_candidates(data: dict[str, Any], n: int = 3) -> list[dict[str, Any]]:
    candidates = data.get("all_candidates", [])
    valid = [c for c in candidates if c["scores"]["valid"]]
    invalid = [c for c in candidates if not c["scores"]["valid"]]
    valid.sort(key=lambda c: float(c["scores"]["final_score"]), reverse=True)
    invalid.sort(key=lambda c: float(c["scores"]["final_score"]), reverse=True)
    return (valid + invalid)[:n]


def _contact_points(candidate: dict[str, Any] | None) -> list[np.ndarray]:
    if candidate is None:
        return []
    return [
        np.array(contact["point_xyz_m"], dtype=np.float64)
        for contact in candidate["contacts"]
    ]


def _contact_rms_m(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    pts_a = _contact_points(a)
    pts_b = _contact_points(b)
    if len(pts_a) != 3 or len(pts_b) != 3:
        return None

    perms = [(0, 1, 2), (0, 2, 1)]
    best = float("inf")
    for perm in perms:
        sq = 0.0
        for i, j in enumerate(perm):
            diff = pts_a[i] - pts_b[j]
            sq += float(diff @ diff)
        best = min(best, (sq / 3.0) ** 0.5)
    return best


def _fmt_score(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def _run_once(
    prefix: str,
    out_dir: Path,
    *,
    use_heatmap: bool,
    use_flexibility: bool,
    search_strategy: str,
) -> dict[str, Any]:
    update_grasp_config(prefix, out_dir)

    env = os.environ.copy()
    env["USE_HEATMAP"] = "1" if use_heatmap else "0"
    env["USE_FLEXIBILITY"] = "1" if use_flexibility else "0"
    env["GRASP_SEARCH_STRATEGY"] = search_strategy

    result = subprocess.run(
        [sys.executable, str(CODE_ROOT / "run_grasps.py")],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"run_grasps.py failed for {prefix} "
            f"({'heatmap+flex' if use_heatmap and use_flexibility else 'heatmap-only' if use_heatmap else 'no-heatmap'})\n"
            f"{result.stderr[-1200:]}"
        )

    data = json.loads(SCORES_PATH.read_text())
    if not use_heatmap:
        dest = out_dir / "grasp_scores_no_heatmap.json"
    elif use_flexibility:
        dest = out_dir / "grasp_scores_with_heatmap_flex.json"
        # Backward-compatible alias for existing tooling.
        (out_dir / "grasp_scores_with_heatmap.json").write_text(json.dumps(data, indent=2))
    else:
        dest = out_dir / "grasp_scores_with_heatmap_no_flex.json"
    dest.write_text(json.dumps(data, indent=2))
    return data


def main() -> int:
    source_index = collect_available_sources()
    evaluations = build_object_evaluations(source_index)

    summaries: list[dict[str, Any]] = []
    search_strategy = os.environ.get("GRASP_SEARCH_STRATEGY", "global_pca_baseline").strip().lower()

    print("\nFull reconstruction heatmap ablation")
    print(f"  Search strategy: {search_strategy}")
    print("")

    for evaluation in evaluations:
        primary = evaluation["primary"]
        prefix = evaluation["prefix"]
        if not primary.get("is_primary", False):
            continue

        _, out_dir, source_label = adapt_source(primary)
        print(f"\n[{prefix}] source={source_label}")

        no_hm = _run_once(
            prefix,
            out_dir,
            use_heatmap=False,
            use_flexibility=False,
            search_strategy=search_strategy,
        )
        with_hm_no_flex = _run_once(
            prefix,
            out_dir,
            use_heatmap=True,
            use_flexibility=False,
            search_strategy=search_strategy,
        )
        with_hm = _run_once(
            prefix,
            out_dir,
            use_heatmap=True,
            use_flexibility=True,
            search_strategy=search_strategy,
        )

        best_with = _best_candidate(with_hm)
        best_with_no_flex = _best_candidate(with_hm_no_flex)
        best_no = _best_candidate(no_hm)
        top_with = _top_candidates(with_hm, 3)
        top_with_no_flex = _top_candidates(with_hm_no_flex, 3)
        top_no = _top_candidates(no_hm, 3)
        contact_rms_m = _contact_rms_m(best_with, best_no)
        object_flexible = bool(
            with_hm.get(
                "object_flexible",
                no_hm.get("object_flexible", primary.get("object_flexible", False)),
            )
        )

        summary = {
            "prefix": prefix,
            "source_label": source_label,
            "object_class": evaluation["object_class"],
            "object_flexible": object_flexible,
            "out_dir": str(out_dir),
            "search_strategy": search_strategy,
            "with_heatmap": {
                "mode": best_with["gripper_mode"] if best_with else None,
                "patch_id": best_with["patch_id"] if best_with else None,
                "score": best_with["scores"]["final_score"] if best_with else None,
                "valid": best_with["scores"]["valid"] if best_with else None,
                "heatmap_score": best_with["scores"]["heatmap_score"] if best_with else None,
                "top3": [
                    {
                        "rank": idx + 1,
                        "mode": cand["gripper_mode"],
                        "patch_id": cand["patch_id"],
                        "score": cand["scores"]["final_score"],
                        "valid": cand["scores"]["valid"],
                    }
                    for idx, cand in enumerate(top_with)
                ],
            },
            "with_heatmap_no_flex": {
                "mode": best_with_no_flex["gripper_mode"] if best_with_no_flex else None,
                "patch_id": best_with_no_flex["patch_id"] if best_with_no_flex else None,
                "score": best_with_no_flex["scores"]["final_score"] if best_with_no_flex else None,
                "valid": best_with_no_flex["scores"]["valid"] if best_with_no_flex else None,
                "heatmap_score": best_with_no_flex["scores"]["heatmap_score"] if best_with_no_flex else None,
                "top3": [
                    {
                        "rank": idx + 1,
                        "mode": cand["gripper_mode"],
                        "patch_id": cand["patch_id"],
                        "score": cand["scores"]["final_score"],
                        "valid": cand["scores"]["valid"],
                    }
                    for idx, cand in enumerate(top_with_no_flex)
                ],
            },
            "no_heatmap": {
                "mode": best_no["gripper_mode"] if best_no else None,
                "patch_id": best_no["patch_id"] if best_no else None,
                "score": best_no["scores"]["final_score"] if best_no else None,
                "valid": best_no["scores"]["valid"] if best_no else None,
                "heatmap_score": best_no["scores"]["heatmap_score"] if best_no else None,
                "top3": [
                    {
                        "rank": idx + 1,
                        "mode": cand["gripper_mode"],
                        "patch_id": cand["patch_id"],
                        "score": cand["scores"]["final_score"],
                        "valid": cand["scores"]["valid"],
                    }
                    for idx, cand in enumerate(top_no)
                ],
            },
            "same_mode": (
                bool(best_with and best_no and best_with["gripper_mode"] == best_no["gripper_mode"])
            ),
            "same_patch": (
                bool(best_with and best_no and best_with["patch_id"] == best_no["patch_id"])
            ),
            "contact_rms_m": round(contact_rms_m, 6) if contact_rms_m is not None else None,
            "contact_rms_cm": round(contact_rms_m * 100.0, 3) if contact_rms_m is not None else None,
        }
        summaries.append(summary)

        print(
            f"  with heatmap+flex : flexible={summary['object_flexible']} "
            f"mode={summary['with_heatmap']['mode']} "
            f"score={_fmt_score(summary['with_heatmap']['score'])} valid={summary['with_heatmap']['valid']}"
        )
        print(
            f"  with heatmap only : mode={summary['with_heatmap_no_flex']['mode']} "
            f"score={_fmt_score(summary['with_heatmap_no_flex']['score'])} valid={summary['with_heatmap_no_flex']['valid']}"
        )
        print(
            f"  no heatmap   : mode={summary['no_heatmap']['mode']} "
            f"score={_fmt_score(summary['no_heatmap']['score'])} valid={summary['no_heatmap']['valid']}"
        )
        print(
            f"  same mode={summary['same_mode']} same patch={summary['same_patch']} "
            f"contact_rms_cm={summary['contact_rms_cm']}"
        )

    report = {
        "search_strategy": search_strategy,
        "object_count": len(summaries),
        "objects": summaries,
    }
    SUMMARY_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved summary to {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
