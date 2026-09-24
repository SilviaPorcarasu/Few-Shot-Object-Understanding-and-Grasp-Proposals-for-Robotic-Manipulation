#!/usr/bin/env python3
"""
Render side-by-side grasp comparison images:

  LEFT  = no heatmap
  RIGHT = with heatmap

The script keeps symmetric and asymmetric primary cases separate by simply
working per adapted case directory under grasp_ready_objects/<object>/<case>/.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "refactored_sam_pipeline-3"
DEFAULT_ROOT = DEFAULT_OUTPUT_ROOT / "grasp_ready_objects"
DEFAULT_HEATMAP_DIR = DEFAULT_OUTPUT_ROOT / "grasp_ready_heatmaps"


def _case_dirs(root: Path) -> list[Path]:
    cases = []
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


def _has_best_candidate(scores_path: Path) -> bool:
    data = json.loads(scores_path.read_text())
    return data.get("best_candidate") is not None


def _run_visualize(
    case_dir: Path,
    prefix: str,
    *,
    use_heatmap: bool,
    scores_path: Path,
    output_png: Path,
    label: str,
    heatmap_dir: Path,
) -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["GRASP_FRAME_DIR"] = str(case_dir)
    env["GRASP_FRAME_PREFIX"] = prefix
    env["GRASP_HEATMAP_DIR"] = str(heatmap_dir)
    env["GRASP_SCORES_PATH"] = str(scores_path)
    env["GRASP_OUTPUT_PNG"] = str(output_png)
    env["GRASP_VIS_LABEL"] = label
    env["USE_HEATMAP"] = "1" if use_heatmap else "0"

    result = subprocess.run(
        [sys.executable, str(CODE_ROOT / "visualize_grasp.py")],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"visualize_grasp.py failed for {case_dir} ({label})\n"
            f"{result.stderr[-1200:]}"
        )


def _placeholder_image(
    title: str,
    subtitle: str,
    size: tuple[int, int] = (1600, 900),
) -> Image.Image:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font_big = ImageFont.load_default()
    font_small = ImageFont.load_default()
    x = 60
    y = 80
    draw.text((x, y), title, fill="black", font=font_big)
    draw.text((x, y + 40), subtitle, fill="dimgray", font=font_small)
    return img


def _compose_side_by_side(
    left_img: Image.Image,
    right_img: Image.Image,
    *,
    header: str,
) -> Image.Image:
    pad = 30
    header_h = 80
    width = left_img.width + right_img.width + 3 * pad
    height = max(left_img.height, right_img.height) + 2 * pad + header_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 20), header, fill="black", font=font)
    canvas.paste(left_img, (pad, header_h + pad))
    canvas.paste(right_img, (2 * pad + left_img.width, header_h + pad))
    return canvas


def render_case(case_dir: Path, heatmap_dir: Path) -> Path:
    prefix = _prefix_from_case(case_dir)
    no_hm_scores = case_dir / "grasp_scores_no_heatmap.json"
    with_hm_scores = case_dir / "grasp_scores_with_heatmap.json"

    left_tmp = case_dir / "_tmp_no_heatmap.png"
    right_tmp = case_dir / "_tmp_with_heatmap.png"
    out_png = case_dir / f"grasp_side_by_side_{prefix}_{case_dir.name}.png"
    out_pdf = out_png.with_suffix(".pdf")

    left_ok = _has_best_candidate(no_hm_scores)
    right_ok = _has_best_candidate(with_hm_scores)

    if left_ok:
        _run_visualize(
            case_dir,
            prefix,
            use_heatmap=False,
            scores_path=no_hm_scores,
            output_png=left_tmp,
            label="No heatmap",
            heatmap_dir=heatmap_dir,
        )
        left_img = Image.open(left_tmp).convert("RGB")
    else:
        left_img = _placeholder_image(
            "No heatmap",
            "No grasp candidate was generated for this case.",
        )

    if right_ok:
        _run_visualize(
            case_dir,
            prefix,
            use_heatmap=True,
            scores_path=with_hm_scores,
            output_png=right_tmp,
            label="With heatmap",
            heatmap_dir=heatmap_dir,
        )
        right_img = Image.open(right_tmp).convert("RGB")
    else:
        right_img = _placeholder_image(
            "With heatmap",
            "No grasp candidate was generated for this case.",
        )

    header = f"{prefix} — {case_dir.name} — left: no heatmap / right: with heatmap"
    composed = _compose_side_by_side(left_img, right_img, header=header)
    composed.save(out_png)
    composed.save(out_pdf, "PDF", resolution=140.0)

    for tmp in (left_tmp, right_tmp):
        if tmp.exists():
            tmp.unlink()
    return out_png


def main() -> int:
    root = Path(os.environ.get("GRASP_READY_ROOT", str(DEFAULT_ROOT))).expanduser()
    heatmap_dir = Path(
        os.environ.get(
            "GRASP_HEATMAP_DIR",
            str(root.parent / "grasp_ready_heatmaps" if root.name == "grasp_ready_objects" else DEFAULT_HEATMAP_DIR),
        )
    ).expanduser()
    cases = _case_dirs(root)
    if not cases:
        raise RuntimeError(f"No primary case directories with both score files found under {root}")

    print("\nRendering side-by-side heatmap comparisons")
    for case_dir in cases:
        out = render_case(case_dir, heatmap_dir)
        print(f"  [save] {out}")
        print(f"  [save] {out.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
