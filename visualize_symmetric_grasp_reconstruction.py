#!/usr/bin/env python3
"""
Visualize symmetric grasp reconstruction outputs.

Expected input:
  - one object folder produced by run_symmetric_grasp_reconstruction.py
    containing:
      object_cloud.ply
      symmetry_axis.ply          optional
      patch_XX_original.ply
      patch_XX_symmetric.ply
      metadata.json              optional

Examples PowerShell:

  python visualize_symmetric_grasp_reconstruction.py `
    --object-dir outputs_symmetric_3d\blue_cup

  python visualize_symmetric_grasp_reconstruction.py `
    --object-dir outputs_symmetric_3d\blue_cup `
    --patch-id 1

  python visualize_symmetric_grasp_reconstruction.py `
    --object-dir outputs_symmetric_3d\roll_of_tape `
    --save-fig outputs_symmetric_3d\roll_of_tape\viz.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def read_ascii_ply(path: Path) -> np.ndarray:
    """
    Minimal ASCII PLY reader for point clouds with x y z as first 3 properties.
    Works with the simple .ply files saved by the reconstruction script.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines or not lines[0].strip().lower().startswith("ply"):
        raise ValueError(f"Not a PLY file: {path}")

    vertex_count = None
    header_end = None

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped.startswith("element vertex"):
            vertex_count = int(line_stripped.split()[-1])
        if line_stripped == "end_header":
            header_end = i + 1
            break

    if vertex_count is None or header_end is None:
        raise ValueError(f"Invalid PLY header: {path}")

    pts = []
    for line in lines[header_end: header_end + vertex_count]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float64)

    return np.asarray(pts, dtype=np.float64)


def load_metadata(object_dir: Path) -> dict:
    meta_path = object_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_patch_files(object_dir: Path, patch_id: Optional[int]) -> Tuple[Path, Path]:
    if patch_id is not None:
        original = object_dir / f"patch_{patch_id:02d}_original.ply"
        symmetric = object_dir / f"patch_{patch_id:02d}_symmetric.ply"
        if not original.exists() or not symmetric.exists():
            raise FileNotFoundError(
                f"Could not find patch {patch_id:02d}. Expected:\n"
                f"  {original}\n"
                f"  {symmetric}"
            )
        return original, symmetric

    originals = sorted(object_dir.glob("patch_*_original.ply"))
    if not originals:
        raise FileNotFoundError(f"No patch_*_original.ply files found in {object_dir}")

    original = originals[0]
    symmetric = object_dir / original.name.replace("_original.ply", "_symmetric.ply")
    if not symmetric.exists():
        raise FileNotFoundError(f"Missing symmetric file for {original.name}: {symmetric}")

    return original, symmetric


def infer_axis_from_metadata(meta: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns axis_point, axis_direction if metadata contains recognizable fields.
    The reconstruction script may save names such as:
      axis_point_3d, axis_direction_3d
      symmetry_axis_point, symmetry_axis_direction
      axis_point, axis_direction
    """
    point_keys = ["axis_point_3d", "symmetry_axis_point_3d", "symmetry_axis_point", "axis_point"]
    dir_keys = ["axis_direction_3d", "symmetry_axis_direction_3d", "symmetry_axis_direction", "axis_direction"]

    p = None
    v = None

    for k in point_keys:
        if k in meta:
            p = np.asarray(meta[k], dtype=np.float64)
            break

    for k in dir_keys:
        if k in meta:
            v = np.asarray(meta[k], dtype=np.float64)
            break

    if p is None or v is None or p.shape[0] != 3 or v.shape[0] != 3:
        return None, None

    n = np.linalg.norm(v)
    if n < 1e-9:
        return None, None

    return p, v / n


def infer_axis_from_axis_ply(axis_ply: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not axis_ply.exists():
        return None, None

    pts = read_ascii_ply(axis_ply)
    if pts.shape[0] < 2:
        return None, None

    p0 = pts[0]
    p1 = pts[-1]
    v = p1 - p0
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None, None

    return p0, v / n


def fallback_axis_from_cloud(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback only for visualization: robust-ish center + main PCA direction.
    This is NOT used as reconstruction logic, only to draw something if metadata is missing.
    """
    center = np.median(points, axis=0)
    x = points - center
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    direction = vh[0]
    return center, direction / np.linalg.norm(direction)


def axis_segment(axis_point: np.ndarray, axis_dir: np.ndarray, points: np.ndarray, scale: float = 1.25):
    centered = points - axis_point
    t = centered @ axis_dir
    t_min, t_max = np.percentile(t, [1, 99])
    mid = 0.5 * (t_min + t_max)
    half = 0.5 * (t_max - t_min) * scale
    a = axis_point + (mid - half) * axis_dir
    b = axis_point + (mid + half) * axis_dir
    return a, b


def set_axes_equal(ax, points: np.ndarray):
    """Make 3D axes have equal scale."""
    if points.size == 0:
        return

    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    centers = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    if not np.isfinite(radius) or radius <= 0:
        radius = 0.05

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points).astype(int)
    return points[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-dir", required=True, help="Folder for one reconstructed symmetric object.")
    parser.add_argument("--patch-id", type=int, default=None, help="Patch number, e.g. 1 for patch_01. Defaults to first patch.")
    parser.add_argument("--max-object-points", type=int, default=15000)
    parser.add_argument("--save-fig", default=None, help="Optional path to save PNG figure.")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive matplotlib window.")
    parser.add_argument("--elev", type=float, default=22.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    args = parser.parse_args()

    object_dir = Path(args.object_dir)
    if not object_dir.exists():
        raise FileNotFoundError(f"Object folder not found: {object_dir}")

    meta = load_metadata(object_dir)

    object_cloud = read_ascii_ply(object_dir / "object_cloud.ply")
    original_file, symmetric_file = find_patch_files(object_dir, args.patch_id)
    original_patch = read_ascii_ply(original_file)
    symmetric_patch = read_ascii_ply(symmetric_file)

    axis_point, axis_dir = infer_axis_from_metadata(meta)
    if axis_point is None or axis_dir is None:
        axis_point, axis_dir = infer_axis_from_axis_ply(object_dir / "symmetry_axis.ply")
    if axis_point is None or axis_dir is None:
        axis_point, axis_dir = fallback_axis_from_cloud(object_cloud)

    object_plot = downsample(object_cloud, args.max_object_points)

    all_points = np.vstack([
        object_plot,
        original_patch if original_patch.size else np.empty((0, 3)),
        symmetric_patch if symmetric_patch.size else np.empty((0, 3)),
    ])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    label = meta.get("label", object_dir.name)
    patch_name = original_file.name.replace("_original.ply", "")

    ax.scatter(
        object_plot[:, 0], object_plot[:, 1], object_plot[:, 2],
        s=2, alpha=0.08, c="gray", label="object pointcloud"
    )

    if original_patch.shape[0] > 0:
        ax.scatter(
            original_patch[:, 0], original_patch[:, 1], original_patch[:, 2],
            s=12, alpha=0.90, c="limegreen", label="original grasp patch"
        )

    if symmetric_patch.shape[0] > 0:
        ax.scatter(
            symmetric_patch[:, 0], symmetric_patch[:, 1], symmetric_patch[:, 2],
            s=12, alpha=0.90, c="cyan", label="symmetric grasp patch"
        )

    # Centroids and distance vector
    if original_patch.shape[0] > 0 and symmetric_patch.shape[0] > 0:
        c1 = original_patch.mean(axis=0)
        c2 = symmetric_patch.mean(axis=0)
        dist = float(np.linalg.norm(c2 - c1))

        ax.scatter([c1[0]], [c1[1]], [c1[2]], s=180, c="red", edgecolors="black", label="original centroid")
        ax.scatter([c2[0]], [c2[1]], [c2[2]], s=180, c="magenta", marker="X", edgecolors="black", label="symmetric centroid")
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]], linewidth=3, c="dodgerblue", label=f"centroid distance = {dist:.4f} m")

    # Symmetry/separation axis
    a, b = axis_segment(axis_point, axis_dir, object_cloud)
    ax.plot(
        [a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
        linewidth=5, c="orange", label="estimated symmetry axis"
    )
    ax.scatter([axis_point[0]], [axis_point[1]], [axis_point[2]], s=120, c="orange", marker="*", edgecolors="black", label="axis point")

    # Optional camera origin
    ax.scatter([0], [0], [0], s=140, c="black", marker="x", label="camera origin")

    ax.set_title(f"{label}: {patch_name} | object + original patch + symmetric patch + axis", fontsize=16)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    set_axes_equal(ax, all_points)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.legend(loc="upper right")

    plt.tight_layout()

    if args.save_fig:
        save_path = Path(args.save_fig)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180)
        print(f"Saved figure: {save_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
