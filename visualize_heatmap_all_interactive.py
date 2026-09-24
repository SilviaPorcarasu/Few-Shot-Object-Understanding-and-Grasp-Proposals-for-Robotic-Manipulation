#!/usr/bin/env python3
"""
Open the interactive heatmap comparison viewer for every primary case.

The windows are shown one after another. Close the current window to move to
the next object/case.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_ROOT = PROJECT_ROOT / "refactored_sam_pipeline-2" / "grasp_ready_objects"


def _case_dirs(root: Path) -> list[Path]:
    cases: list[Path] = []
    for case_dir in sorted(root.glob("*/*")):
        if not case_dir.is_dir():
            continue
        if not case_dir.name.startswith("primary_"):
            continue
        if (case_dir / "grasp_scores_no_heatmap.json").exists() and (case_dir / "grasp_scores_with_heatmap.json").exists():
            cases.append(case_dir)
    return cases


def _prefix_from_case(case_dir: Path) -> str:
    object_cloud = next(case_dir.glob("*_object_cloud.ply"), None)
    if object_cloud is not None:
        return object_cloud.name.replace("_object_cloud.ply", "")
    scores_path = case_dir / "grasp_scores_no_heatmap.json"
    if scores_path.exists():
        data = json.loads(scores_path.read_text())
        if data.get("frame_prefix"):
            return str(data["frame_prefix"])
    return case_dir.parent.name


def main() -> int:
    root = Path(os.environ.get("GRASP_READY_ROOT", str(DEFAULT_ROOT))).expanduser()
    cases = _case_dirs(root)
    if not cases:
        raise RuntimeError(f"No primary cases with comparison scores found under {root}")

    print("\nOpening interactive heatmap comparisons")
    print("Close the current figure to continue to the next case.\n")

    base_env = os.environ.copy()
    base_env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grasp-matplotlib"))
    base_env.pop("MPLBACKEND", None)

    total = len(cases)
    for idx, case_dir in enumerate(cases, start=1):
        prefix = _prefix_from_case(case_dir)
        print(f"[{idx}/{total}] {prefix} — {case_dir.name}")
        result = subprocess.run(
            [
                sys.executable,
                str(CODE_ROOT / "visualize_heatmap_side_by_side.py"),
                "--case-dir",
                str(case_dir),
            ],
            cwd=str(PROJECT_ROOT),
            env=base_env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"visualize_heatmap_side_by_side.py failed for {case_dir} "
                f"(exit {result.returncode})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
