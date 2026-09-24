#!/usr/bin/env python3
"""
Interactive top-K side-by-side grasp comparison.

Each row shows the same grasp rank:
  LEFT  = no heatmap
  RIGHT = with heatmap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from grasp_config import FRAME_DIR as DEFAULT_FRAME_DIR, HEATMAP_DIR as DEFAULT_HEATMAP_DIR
from visualize_grasp import _set_padded_limits
from visualize_heatmap_side_by_side import (
    _attach_scroll_zoom,
    _attach_view_sync,
    _infer_prefix,
    _load_case_data,
    _plot_case,
    _resolve_scores_path,
)


def _rank_candidates(report: dict, topk: int) -> list[dict]:
    candidates = report.get("all_candidates", [])
    valid = [c for c in candidates if c["scores"]["valid"]]
    invalid = [c for c in candidates if not c["scores"]["valid"]]
    valid.sort(key=lambda c: float(c["scores"]["final_score"]), reverse=True)
    invalid.sort(key=lambda c: float(c["scores"]["final_score"]), reverse=True)
    return (valid + invalid)[:topk]


def _with_best(data: dict, candidate: dict | None) -> dict:
    clone = dict(data)
    clone["best"] = candidate
    return clone


def main() -> int:
    parser = argparse.ArgumentParser(description="Show top-K no-heatmap vs heatmap grasps side by side.")
    parser.add_argument("--case-dir", default=str(DEFAULT_FRAME_DIR), help="Grasp-ready case directory.")
    parser.add_argument("--prefix", help="Object prefix. Inferred when omitted.")
    parser.add_argument("--topk", type=int, default=3, help="How many grasps to show per side.")
    parser.add_argument("--output", help="Optional PNG output path.")
    args = parser.parse_args()

    case_dir = Path(args.case_dir).expanduser()
    prefix = args.prefix or _infer_prefix(case_dir)
    no_scores = _resolve_scores_path(case_dir, "grasp_scores_no_heatmap.json")
    with_scores = _resolve_scores_path(case_dir, "grasp_scores_with_heatmap.json")
    if not no_scores.exists() or not with_scores.exists():
        raise RuntimeError(
            "Missing comparison score files. Expected both "
            f"{no_scores.name} and {with_scores.name} in {case_dir}"
        )

    no_data = _load_case_data(case_dir, prefix, no_scores, use_heatmap=False, heatmap_dir=DEFAULT_HEATMAP_DIR)
    with_data = _load_case_data(case_dir, prefix, with_scores, use_heatmap=True, heatmap_dir=DEFAULT_HEATMAP_DIR)
    no_ranked = _rank_candidates(no_data["report"], args.topk)
    with_ranked = _rank_candidates(with_data["report"], args.topk)

    rows = max(len(no_ranked), len(with_ranked), 1)
    fig = plt.figure(figsize=(18, 7 * rows))
    fig.patch.set_facecolor("white")
    axes = []

    shared_bounds = np.vstack([
        no_data["obj_vis_pts"],
        with_data["obj_vis_pts"],
    ])

    for row in range(rows):
        ax_left = fig.add_subplot(rows, 2, row * 2 + 1, projection="3d")
        ax_right = fig.add_subplot(rows, 2, row * 2 + 2, projection="3d")
        ax_left.set_facecolor("white")
        ax_right.set_facecolor("white")
        axes.extend([ax_left, ax_right])

        left_best = no_ranked[row] if row < len(no_ranked) else None
        right_best = with_ranked[row] if row < len(with_ranked) else None

        _plot_case(
            ax_left,
            _with_best(no_data, left_best),
            panel_label=f"No heatmap — Top {row + 1}",
            show_legend=False,
        )
        _plot_case(
            ax_right,
            _with_best(with_data, right_best),
            panel_label=f"With heatmap — Top {row + 1}",
            show_legend=(row == 0),
        )

        _set_padded_limits(ax_left, shared_bounds, pad_ratio=0.14, min_pad=0.01)
        _set_padded_limits(ax_right, shared_bounds, pad_ratio=0.14, min_pad=0.01)
        ax_left.view_init(elev=18, azim=-35)
        ax_right.view_init(elev=18, azim=-35)

    fig.suptitle(
        f"{prefix} — top {rows} grasps comparison  ·  left=no heatmap / right=with heatmap",
        fontsize=13,
        weight="bold",
        y=0.995,
    )
    _attach_scroll_zoom(fig)
    _attach_view_sync(fig, axes)
    plt.tight_layout()

    output_png = (
        Path(args.output).expanduser()
        if args.output
        else case_dir / f"grasp_top{rows}_compare_{prefix}_{case_dir.name}.png"
    )
    fig.savefig(output_png, dpi=140, bbox_inches="tight")
    print(f"[save] {output_png}")

    if "agg" not in plt.get_backend().lower():
        try:
            plt.show()
        except Exception as exc:
            print(f"[warn] interactive display failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
