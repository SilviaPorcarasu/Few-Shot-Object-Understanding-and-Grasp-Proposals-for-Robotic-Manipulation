#!/usr/bin/env python3
"""
Central entrypoint for the grasp_work pipeline around the current SAM3 stack.

This script does not re-implement grasping. It keeps the existing modules
separate, but gives them one clean place from which they can be launched.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "grasp_pipeline"
TMP_MPL_DIR = Path(tempfile.gettempdir()) / "grasp-matplotlib"

LEGACY_PATHS = (
    "refactored_sam_pipeline",
    "processed-2",
    "segmantation-heatpred_with_results",
    "external_saved_refactored",
    "__pycache__",
)


def run_python_script(
    script_name: str,
    *,
    script_args: list[str] | None = None,
    env_updates: dict[str, str] | None = None,
) -> None:
    """Run one local Python script with optional environment overrides."""
    cmd = [sys.executable, str(CODE_DIR / script_name)]
    if script_args:
        cmd.extend(script_args)

    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)

    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")


def build_matplotlib_env(use_agg: bool) -> dict[str, str]:
    """Use a writable matplotlib cache, optionally forcing PNG-only mode."""
    env = {"MPLCONFIGDIR": str(TMP_MPL_DIR)}
    if use_agg:
        env["MPLBACKEND"] = "Agg"
    return env


def cleanup_legacy_data() -> list[Path]:
    """Remove legacy data folders that are not part of the current SAM3 flow."""
    removed: list[Path] = []
    for rel_path in LEGACY_PATHS:
        path = ROOT / rel_path
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)
    return removed


def cmd_cleanup(_args: argparse.Namespace) -> int:
    removed = cleanup_legacy_data()
    if removed:
        print("Removed legacy paths:")
        for path in removed:
            print(f"  - {path}")
    else:
        print("No legacy paths were present.")
    return 0


def cmd_grasp_all(args: argparse.Namespace) -> int:
    if args.cleanup_legacy:
        removed = cleanup_legacy_data()
        if removed:
            print("Removed legacy paths before grasp run:")
            for path in removed:
                print(f"  - {path}")

    env = build_matplotlib_env(args.agg)
    env["GRASP_EVAL_MODE"] = args.mode
    env["ASYMMETRIC_CAMERA_SOURCE"] = args.camera_source
    env["WRENCH"] = "1" if args.wrench else "0"
    env["GRASP_SEARCH_STRATEGY"] = args.search_strategy

    if args.pipeline_root:
        env["SAM_PIPELINE_ROOT"] = str(Path(args.pipeline_root).expanduser().resolve())
    if args.pipeline_outputs:
        env["SAM_PIPELINE_OUTPUTS"] = str(Path(args.pipeline_outputs).expanduser().resolve())
    if args.output_root:
        env["GRASP_OUTPUT_ROOT"] = str(Path(args.output_root).expanduser().resolve())

    run_python_script(
        "run_grasps_on_sam3_outputs.py",
        env_updates=env,
    )
    return 0


def cmd_compare_heatmap(args: argparse.Namespace) -> int:
    env = build_matplotlib_env(args.agg)
    env["GRASP_SEARCH_STRATEGY"] = args.search_strategy
    run_python_script(
        "compare_heatmap.py",
        env_updates=env,
    )
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    run_python_script(
        "visualize_grasp.py",
        env_updates=build_matplotlib_env(args.agg),
    )
    return 0


def cmd_thesis_figures(args: argparse.Namespace) -> int:
    script_args: list[str] = []
    if args.case_dir:
        script_args.extend(["--case-dir", args.case_dir])
    if args.output_dir:
        script_args.extend(["--output-dir", args.output_dir])

    run_python_script(
        "export_thesis_grasp_stage_figures.py",
        script_args=script_args,
        env_updates=build_matplotlib_env(True),
    )
    return 0


def cmd_compare_reference(args: argparse.Namespace) -> int:
    script_args = ["--reference", args.reference]
    if args.result:
        script_args.extend(["--result", args.result])
    if args.output:
        script_args.extend(["--output", args.output])
    if args.include_invalid:
        script_args.append("--include-invalid")

    run_python_script(
        "benchmark_reference_grasps.py",
        script_args=script_args,
        env_updates=build_matplotlib_env(True),
    )
    return 0


def cmd_compare_baseline(args: argparse.Namespace) -> int:
    script_args: list[str] = []
    if args.result:
        script_args.extend(["--result", args.result])
    if args.output:
        script_args.extend(["--output", args.output])

    run_python_script(
        "compare_global_baseline.py",
        script_args=script_args,
        env_updates=build_matplotlib_env(True),
    )
    return 0


def cmd_compare_heatmap_all(args: argparse.Namespace) -> int:
    env = build_matplotlib_env(True)
    env["GRASP_SEARCH_STRATEGY"] = args.search_strategy
    run_python_script(
        "compare_heatmap_all_primary.py",
        env_updates=env,
    )
    return 0


def cmd_render_heatmap_side_by_side(_args: argparse.Namespace) -> int:
    run_python_script(
        "render_heatmap_side_by_side.py",
        env_updates=build_matplotlib_env(True),
    )
    return 0


def cmd_visualize_heatmap_side_by_side(args: argparse.Namespace) -> int:
    script_args: list[str] = []
    if args.case_dir:
        script_args.extend(["--case-dir", args.case_dir])
    if args.prefix:
        script_args.extend(["--prefix", args.prefix])
    if args.output:
        script_args.extend(["--output", args.output])

    run_python_script(
        "visualize_heatmap_side_by_side.py",
        script_args=script_args,
        env_updates=build_matplotlib_env(args.agg),
    )
    return 0


def cmd_visualize_heatmap_all_interactive(_args: argparse.Namespace) -> int:
    run_python_script(
        "visualize_heatmap_all_interactive.py",
        env_updates=build_matplotlib_env(False),
    )
    return 0


def cmd_visualize_heatmap_topk_side_by_side(args: argparse.Namespace) -> int:
    script_args: list[str] = []
    if args.case_dir:
        script_args.extend(["--case-dir", args.case_dir])
    if args.prefix:
        script_args.extend(["--prefix", args.prefix])
    script_args.extend(["--topk", str(args.topk)])
    if args.output:
        script_args.extend(["--output", args.output])

    run_python_script(
        "visualize_heatmap_topk_side_by_side.py",
        script_args=script_args,
        env_updates=build_matplotlib_env(args.agg),
    )
    return 0


def cmd_visualize_heatmap_topk_all_interactive(args: argparse.Namespace) -> int:
    env = build_matplotlib_env(False)
    env["GRASP_TOPK"] = str(args.topk)
    run_python_script(
        "visualize_heatmap_topk_all_interactive.py",
        env_updates=env,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified launcher for grasp_work and the current SAM3 pipeline."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cleanup = sub.add_parser(
        "cleanup-legacy",
        help="Delete old non-SAM3 data folders.",
    )
    cleanup.set_defaults(func=cmd_cleanup)

    grasp_all = sub.add_parser(
        "grasp-all",
        help="Adapt SAM3 outputs and run grasping for every available object.",
    )
    grasp_all.add_argument(
        "--mode",
        choices=("primary_only", "compare_views"),
        default="primary_only",
        help="How to evaluate available sources from the current SAM outputs.",
    )
    grasp_all.add_argument(
        "--camera-source",
        choices=("view1", "view2", "midpoint"),
        default="view1",
        help="Camera origin used for asymmetric fused objects.",
    )
    grasp_all.add_argument(
        "--pipeline-root",
        help="Optional override for the SAM pipeline root.",
    )
    grasp_all.add_argument(
        "--pipeline-outputs",
        help="Optional override for the chosen outputs_full_pipeline* folder.",
    )
    grasp_all.add_argument(
        "--output-root",
        help="Optional override for where grasp_ready_objects will be written.",
    )
    grasp_all.add_argument(
        "--wrench",
        action="store_true",
        help="Enable the optional wrench score.",
    )
    grasp_all.add_argument(
        "--search-strategy",
        choices=("global_pca_baseline", "local_patches"),
        default="global_pca_baseline",
        help="Analytical candidate-search strategy used before optional heatmap priors.",
    )
    grasp_all.add_argument(
        "--agg",
        action="store_true",
        help="Force non-interactive matplotlib output.",
    )
    grasp_all.add_argument(
        "--cleanup-legacy",
        action="store_true",
        help="Delete old non-SAM3 data folders before the run.",
    )
    grasp_all.set_defaults(func=cmd_grasp_all)

    compare = sub.add_parser(
        "compare-heatmap",
        help="Run the current configured case once with heatmap and once without.",
    )
    compare.add_argument(
        "--agg",
        action="store_true",
        help="Force non-interactive matplotlib output.",
    )
    compare.add_argument(
        "--search-strategy",
        choices=("global_pca_baseline", "local_patches"),
        default="global_pca_baseline",
        help="Search strategy used for both heatmap/no-heatmap runs.",
    )
    compare.set_defaults(func=cmd_compare_heatmap)

    visualize = sub.add_parser(
        "visualize",
        help="Render the current configured grasp visualization.",
    )
    visualize.add_argument(
        "--agg",
        action="store_true",
        help="Force non-interactive matplotlib output.",
    )
    visualize.set_defaults(func=cmd_visualize)

    thesis = sub.add_parser(
        "thesis-figures",
        help="Export the 3 thesis stage figures for one grasp-ready case.",
    )
    thesis.add_argument(
        "--case-dir",
        help="Path to one grasp-ready case directory.",
    )
    thesis.add_argument(
        "--output-dir",
        help="Directory where the stage figures should be saved.",
    )
    thesis.set_defaults(func=cmd_thesis_figures)

    compare_reference = sub.add_parser(
        "compare-reference",
        help="Compare one local grasp result JSON against external reference grasps.",
    )
    compare_reference.add_argument(
        "--reference",
        required=True,
        help="Path to reference grasp JSON exported from GraspIt!/paper/manual annotation.",
    )
    compare_reference.add_argument(
        "--result",
        help="Path to local result JSON. Defaults to grasp_scores_no_heatmap.json, then grasp_scores.json.",
    )
    compare_reference.add_argument(
        "--output",
        help="Optional output JSON path.",
    )
    compare_reference.add_argument(
        "--include-invalid",
        action="store_true",
        help="Allow matching against invalid local candidates too.",
    )
    compare_reference.set_defaults(func=cmd_compare_reference)

    compare_baseline = sub.add_parser(
        "compare-baseline",
        help="Compare the current no-heatmap result against a simple global-PCA baseline.",
    )
    compare_baseline.add_argument(
        "--result",
        help="Path to local result JSON. Defaults to grasp_scores_no_heatmap.json, then grasp_scores.json.",
    )
    compare_baseline.add_argument(
        "--output",
        help="Optional output JSON path.",
    )
    compare_baseline.set_defaults(func=cmd_compare_baseline)

    compare_all = sub.add_parser(
        "compare-heatmap-all",
        help="Run no-heatmap vs heatmap over all primary reconstructed objects.",
    )
    compare_all.add_argument(
        "--search-strategy",
        choices=("global_pca_baseline", "local_patches"),
        default="global_pca_baseline",
        help="Search strategy used for all objects in the ablation.",
    )
    compare_all.set_defaults(func=cmd_compare_heatmap_all)

    render_compare = sub.add_parser(
        "render-heatmap-side-by-side",
        help="Render left=no-heatmap / right=with-heatmap comparison images for primary cases.",
    )
    render_compare.set_defaults(func=cmd_render_heatmap_side_by_side)

    interactive_compare = sub.add_parser(
        "visualize-heatmap-side-by-side",
        help="Open an interactive left=no-heatmap / right=with-heatmap 3D comparison for one case.",
    )
    interactive_compare.add_argument(
        "--case-dir",
        help="Path to one grasp-ready case directory. Defaults to the currently configured FRAME_DIR.",
    )
    interactive_compare.add_argument(
        "--prefix",
        help="Optional object prefix override. Usually inferred automatically.",
    )
    interactive_compare.add_argument(
        "--output",
        help="Optional PNG output path for the comparison figure.",
    )
    interactive_compare.add_argument(
        "--agg",
        action="store_true",
        help="Force non-interactive matplotlib output.",
    )
    interactive_compare.set_defaults(func=cmd_visualize_heatmap_side_by_side)

    interactive_all = sub.add_parser(
        "visualize-heatmap-all-interactive",
        help="Open the interactive left=no-heatmap / right=with-heatmap viewer for all primary cases, one after another.",
    )
    interactive_all.set_defaults(func=cmd_visualize_heatmap_all_interactive)

    interactive_topk = sub.add_parser(
        "visualize-heatmap-topk-side-by-side",
        help="Open a side-by-side viewer for the top-K no-heatmap vs heatmap grasps.",
    )
    interactive_topk.add_argument(
        "--case-dir",
        help="Path to one grasp-ready case directory. Defaults to the currently configured FRAME_DIR.",
    )
    interactive_topk.add_argument(
        "--prefix",
        help="Optional object prefix override. Usually inferred automatically.",
    )
    interactive_topk.add_argument(
        "--topk",
        type=int,
        default=3,
        help="How many grasps to show per side.",
    )
    interactive_topk.add_argument(
        "--output",
        help="Optional PNG output path for the comparison figure.",
    )
    interactive_topk.add_argument(
        "--agg",
        action="store_true",
        help="Force non-interactive matplotlib output.",
    )
    interactive_topk.set_defaults(func=cmd_visualize_heatmap_topk_side_by_side)

    interactive_topk_all = sub.add_parser(
        "visualize-heatmap-topk-all-interactive",
        help="Open the interactive top-K no-heatmap vs heatmap viewer for all primary cases.",
    )
    interactive_topk_all.add_argument(
        "--topk",
        type=int,
        default=3,
        help="How many grasps to show per side.",
    )
    interactive_topk_all.set_defaults(func=cmd_visualize_heatmap_topk_all_interactive)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
