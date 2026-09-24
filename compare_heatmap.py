#!/usr/bin/env python3
"""
Rulează pipeline-ul de două ori — cu și fără heatmap — și compară
top-N grasps pentru a vedea dacă heatmap-ul schimbă selecția.

Output:
    grasp_scores_with_heatmap.json   — rezultate cu heatmap
    grasp_scores_no_heatmap.json     — rezultate fără heatmap (geometrie pură)
    Tabel comparativ în consolă      — top grasps din ambele rulări
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
SCORES_PATH = PROJECT_ROOT / "grasp_scores.json"
WITH_HM_PATH = PROJECT_ROOT / "grasp_scores_with_heatmap.json"
NO_HM_PATH = PROJECT_ROOT / "grasp_scores_no_heatmap.json"
TOP_N = 5


def run_pipeline(use_heatmap: bool) -> Path:
    label = "cu heatmap" if use_heatmap else "fără heatmap"
    print(f"\n{'='*60}")
    print(f"  Rulare {label} ...")
    print(f"{'='*60}")

    env = os.environ.copy()
    env["USE_HEATMAP"] = "1" if use_heatmap else "0"
    case_dir = None
    try:
        from grasp_config import FRAME_DIR
        case_dir = Path(FRAME_DIR)
    except Exception:
        pass

    result = subprocess.run(
        [sys.executable, str(CODE_ROOT / "run_grasps.py")],
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"run_grasps.py a eșuat (exit {result.returncode})")

    dest = WITH_HM_PATH if use_heatmap else NO_HM_PATH
    shutil.copyfile(SCORES_PATH, dest)
    if case_dir is not None and case_dir.exists():
        shutil.copyfile(SCORES_PATH, case_dir / dest.name)
    print(f"  Salvat în {dest.name}")
    return dest


def load_top(path: Path, n: int) -> list[dict]:
    data = json.loads(path.read_text())
    candidates = data.get("all_candidates", [])
    valid = [c for c in candidates if c["scores"]["valid"]]
    rest = [c for c in candidates if not c["scores"]["valid"]]
    ranked = valid + rest
    return ranked[:n]


def fmt_contacts(candidate: dict) -> str:
    pts = [
        f"({c['point_xyz_cm'][0]:.1f},{c['point_xyz_cm'][1]:.1f},{c['point_xyz_cm'][2]:.1f})"
        for c in candidate["contacts"]
    ]
    return "  ".join(pts)


def print_comparison(with_hm: list[dict], no_hm: list[dict]) -> None:
    print(f"\n{'='*60}")
    print("  COMPARAȚIE TOP GRASPS")
    print(f"{'='*60}")

    header = f"{'#':<3}  {'mod':<8}  {'spread mm':<10}  {'scor':<8}  {'valid':<6}  {'heatmap':<8}  {'thickness':<10}"
    sep = "-" * len(header)

    print(f"\n  CU HEATMAP:")
    print(f"  {header}")
    print(f"  {sep}")
    for i, c in enumerate(with_hm, 1):
        s = c["scores"]
        print(f"  {i:<3}  {c['gripper_mode']:<8}  {c['finger_spread_mm']:<10.1f}  "
              f"{s['final_score']:<8.4f}  {str(s['valid']):<6}  "
              f"{s['heatmap_score']:<8.3f}  {s['thickness_score']:<10.3f}")
        print(f"       contacts (cm): {fmt_contacts(c)}")

    print(f"\n  FĂRĂ HEATMAP (geometrie pură):")
    print(f"  {header}")
    print(f"  {sep}")
    for i, c in enumerate(no_hm, 1):
        s = c["scores"]
        print(f"  {i:<3}  {c['gripper_mode']:<8}  {c['finger_spread_mm']:<10.1f}  "
              f"{s['final_score']:<8.4f}  {str(s['valid']):<6}  "
              f"{s['heatmap_score']:<8.3f}  {s['thickness_score']:<10.3f}")
        print(f"       contacts (cm): {fmt_contacts(c)}")

    # Vede dacă primul grasp valid este același patch
    best_hm = next((c for c in with_hm if c["scores"]["valid"]), with_hm[0] if with_hm else None)
    best_no = next((c for c in no_hm if c["scores"]["valid"]), no_hm[0] if no_hm else None)

    print(f"\n  BEST CU HM:   patch={best_hm['patch_id']}  mod={best_hm['gripper_mode']}"
          f"  scor={best_hm['scores']['final_score']:.4f}" if best_hm else "  (niciun grasp)")
    print(f"  BEST FĂRĂ HM: patch={best_no['patch_id']}  mod={best_no['gripper_mode']}"
          f"  scor={best_no['scores']['final_score']:.4f}" if best_no else "  (niciun grasp)")

    if best_hm and best_no:
        same_patch = best_hm["patch_id"] == best_no["patch_id"]
        same_mode = best_hm["gripper_mode"] == best_no["gripper_mode"]
        print(f"\n  Același patch?  {'DA' if same_patch else 'NU — heatmap schimbă zona!'}")
        print(f"  Același mod?    {'DA' if same_mode else 'NU — heatmap schimbă modul!'}")


def main() -> None:
    run_pipeline(use_heatmap=True)
    run_pipeline(use_heatmap=False)

    with_hm = load_top(WITH_HM_PATH, TOP_N)
    no_hm = load_top(NO_HM_PATH, TOP_N)

    print_comparison(with_hm, no_hm)
    print(f"\nFișiere salvate:\n  {WITH_HM_PATH.name}\n  {NO_HM_PATH.name}\n")


if __name__ == "__main__":
    main()
