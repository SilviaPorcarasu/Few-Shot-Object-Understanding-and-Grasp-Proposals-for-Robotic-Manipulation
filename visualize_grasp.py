#!/usr/bin/env python3
"""
3D rotatable visualisation of the best grasp.

The point cloud is rendered in 3D (drag to rotate, scroll to zoom,
shift+drag to pan). The default view is the camera perspective.

Shows:
  - the patch in 3D, coloured by ML heatmap value
  - the full object cloud (faint)
  - 3 contact points (F1, F2, F3) — large markers, always on top
  - contact normals (small arrows)
  - approach arrow (orange) along the chosen grasp approach direction
  - small XYZ axis triad showing camera frame orientation
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from grasp_config import (
    FRAME_DIR as DEFAULT_FRAME_DIR,
    FRAME_PREFIX as DEFAULT_FRAME_PREFIX,
    HEATMAP_DIR as DEFAULT_HEATMAP_DIR,
    USE_HEATMAP,
)
from grasp_helpers import project_sparse_point_heat_to_object

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
FRAME_DIR = Path(os.environ.get("GRASP_FRAME_DIR", str(DEFAULT_FRAME_DIR))).expanduser()
FRAME_PREFIX = os.environ.get("GRASP_FRAME_PREFIX", DEFAULT_FRAME_PREFIX)
HEATMAP_DIR = Path(os.environ.get("GRASP_HEATMAP_DIR", str(DEFAULT_HEATMAP_DIR))).expanduser()
NPZ_PATH = FRAME_DIR / f"{FRAME_PREFIX}_grasp_patch_data.npz"
OBJECT_PLY_PATH = FRAME_DIR / f"{FRAME_PREFIX}_object_cloud.ply"
SCENE_PLY_PATH = FRAME_DIR / f"{FRAME_PREFIX}_scene_cloud.ply"
SCORES_PATH = Path(
    os.environ.get("GRASP_SCORES_PATH", str(PROJECT_ROOT / "grasp_scores.json"))
).expanduser()
OUTPUT_PNG = Path(
    os.environ.get("GRASP_OUTPUT_PNG", str(PROJECT_ROOT / f"grasp_visualization_{FRAME_PREFIX}.png"))
).expanduser()
VIS_LABEL = os.environ.get("GRASP_VIS_LABEL", "").strip()
PAPER_MODE = os.environ.get("GRASP_PAPER_MODE", "0").strip().lower() in ("1", "true", "yes")
SHOW_CONTACTS = os.environ.get("GRASP_SHOW_CONTACTS", "1").strip().lower() not in ("0", "false", "no")
VIEW_ELEV = float(os.environ.get("GRASP_VIEW_ELEV", "18"))
VIEW_AZIM = float(os.environ.get("GRASP_VIEW_AZIM", "-35"))

CONTACT_COLORS = ["red", "lime", "dodgerblue"]
CONTACT_LABELS = ["F1", "F2", "F3"]


def _m_formatter(value: float, _pos: int) -> str:
    return f"{value:.3f}"


def _robust_inlier_mask(points: np.ndarray, keep_percentile: float = 98.5) -> np.ndarray:
    """Trim sparse geometric outliers so the overview frames the real object."""
    if len(points) < 32:
        return np.ones(len(points), dtype=bool)
    ref = np.median(points, axis=0)
    d = np.linalg.norm(points - ref, axis=1)
    thresh = np.percentile(d, keep_percentile)
    mask = d <= max(float(thresh), 1e-6)
    if mask.sum() < max(16, int(0.35 * len(points))):
        return np.ones(len(points), dtype=bool)
    return mask


def _set_padded_limits(
    ax,
    points: np.ndarray,
    *,
    pad_ratio: float,
    min_pad: float,
) -> tuple[np.ndarray, np.ndarray]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, min_pad)
    pads = np.maximum(spans * pad_ratio, min_pad)
    ax.set_xlim(mins[0] - pads[0], maxs[0] + pads[0])
    ax.set_ylim(mins[1] - pads[1], maxs[1] + pads[1])
    ax.set_zlim(mins[2] - pads[2], maxs[2] + pads[2])
    ax.set_box_aspect(tuple((spans + 2.0 * pads).tolist()))
    return mins, maxs


def read_ply(path: Path):
    """Read points from binary PLY."""
    vertex_count = None
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Incomplete PLY header")
            if line.startswith(b"element vertex"):
                vertex_count = int(line.split()[2])
            if line.strip() == b"end_header":
                break
        dtype = np.dtype([
            ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
            ("nx", "<f8"), ("ny", "<f8"), ("nz", "<f8"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])
        data = np.fromfile(f, dtype=dtype, count=vertex_count)
    return np.column_stack((data["x"], data["y"], data["z"]))


def main():
    npz = np.load(NPZ_PATH, allow_pickle=True)
    patch_pts = np.asarray(npz["points"], dtype=np.float64)
    patch_pixels = np.asarray(npz["pixels"], dtype=np.int32)
    frame_name = str(npz["frame_name"])
    point_heat_values = np.asarray(
        npz["point_heat_values"], dtype=np.float64,
    ) if "point_heat_values" in npz.files else None

    obj_pts = read_ply(OBJECT_PLY_PATH)
    scene_pts = read_ply(SCENE_PLY_PATH) if SCENE_PLY_PATH.exists() else np.zeros((0, 3), dtype=np.float64)
    report = json.loads(SCORES_PATH.read_text())
    best = report["best_candidate"]

    contacts = [np.array(c["point_xyz_m"]) for c in best["contacts"]]
    normals = [np.array(c["normal_xyz_unit"]) for c in best["contacts"]]
    approach = np.array(best["approach_direction_xyz_unit"])
    grasp_centre = np.mean(contacts, axis=0)

    # Heatmap (optional). Prefer direct per-point 3D heat when available:
    # it is already aligned with the patch points we render. The 2D image is
    # only a fallback for older cases where we do not have per-point heat.
    heatmap = None
    hm_path = HEATMAP_DIR / f"{frame_name}_pred_gray.png"
    use_point_heat_display = (
        USE_HEATMAP
        and point_heat_values is not None
        and len(point_heat_values) == len(patch_pts)
    )
    if USE_HEATMAP and hm_path.exists() and not use_point_heat_display:
        try:
            from PIL import Image
            heatmap = np.asarray(Image.open(hm_path).convert("L"))
        except ImportError:
            pass

    obj_inlier_mask = _robust_inlier_mask(obj_pts)
    obj_vis_idx = np.flatnonzero(obj_inlier_mask)
    if len(obj_vis_idx) == 0:
        obj_vis_idx = np.arange(len(obj_pts))
    obj_vis_pts = obj_pts[obj_vis_idx]

    obj_step = max(1, len(obj_vis_idx) // 5000)
    patch_step = max(1, len(patch_pts) // 6000)

    fig = plt.figure(figsize=(7.2, 7.0) if PAPER_MODE else (9.5, 8.2))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # Scene context: show only LOCAL neighbors around the object so the
    # overview keeps the object readable. The full scene cloud is still used
    # by collision/support checks elsewhere; this only affects plotting.
    if len(scene_pts) > 0:
        obj_center = obj_vis_pts.mean(axis=0)
        obj_extent = obj_vis_pts.max(axis=0) - obj_vis_pts.min(axis=0)
        local_scene_radius = max(float(np.linalg.norm(obj_extent)) * 1.25, 0.12)
        local_scene_mask = np.linalg.norm(scene_pts - obj_center, axis=1) <= local_scene_radius
        local_scene = scene_pts[local_scene_mask]
        if len(local_scene) == 0:
            local_scene = scene_pts
        s_step = max(1, len(local_scene) // 6000)
        s = local_scene[::s_step]
        ax.scatter(s[:, 0], s[:, 1], s[:, 2],
                   c="lightgrey", s=0.4, alpha=0.18, depthshade=True)

    # Full observed object, coloured by the same heat prior used by grasping.
    o_idx = obj_vis_idx[::obj_step]
    o = obj_pts[o_idx]
    if use_point_heat_display:
        obj_hm_values = project_sparse_point_heat_to_object(
            obj_pts, patch_pts, np.clip(point_heat_values, 0.0, 1.0),
        )
        hm_vals = np.clip(obj_hm_values[o_idx], 0.0, 1.0)
        ax.scatter(o[:, 0], o[:, 1], o[:, 2],
                   c=hm_vals, cmap="jet", s=2.0, alpha=0.85,
                   vmin=0.0, vmax=1.0, depthshade=False)
    elif heatmap is not None:
        p_idx = np.arange(0, len(patch_pts), patch_step)
        p = patch_pts[p_idx]
        rgb_w, rgb_h = 1280, 720
        hm_h, hm_w = heatmap.shape[:2]
        sx, sy = hm_w / rgb_w, hm_h / rgb_h
        scaled = patch_pixels[p_idx].copy()
        scaled[:, 0] = np.clip((scaled[:, 0] * sx).astype(int), 0, hm_w - 1)
        scaled[:, 1] = np.clip((scaled[:, 1] * sy).astype(int), 0, hm_h - 1)
        hm_vals = heatmap[scaled[:, 1], scaled[:, 0]] / 255.0
        ax.scatter(p[:, 0], p[:, 1], p[:, 2],
                   c=hm_vals, cmap="jet", s=2.0, alpha=0.85,
                   vmin=0.0, vmax=1.0, depthshade=False)
    else:
        ax.scatter(o[:, 0], o[:, 1], o[:, 2],
                   c="steelblue", s=1.2, alpha=0.7)

    if SHOW_CONTACTS:
        # Contact triangle (drawn behind the markers)
        tri = np.array(contacts + [contacts[0]])
        ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], "k--",
                linewidth=1.0, alpha=0.5, zorder=15)

        PREGRASP_STANDOFF_M = 0.008   # 8 mm along contact normal = pre-grasp position
        for i, (ct, n) in enumerate(zip(contacts, normals)):
            n_unit = n / max(float(np.linalg.norm(n)), 1e-12)
            label_pos = ct + n_unit * PREGRASP_STANDOFF_M

            # Small black contact-patch ring on the object surface, lying in
            # the tangent plane (perpendicular to the contact normal).
            ring_radius = 0.003   # 3 mm
            ring_n_pts = 32
            trial2 = np.array([1.0, 0.0, 0.0])
            if abs(trial2 @ n_unit) > 0.9:
                trial2 = np.array([0.0, 1.0, 0.0])
            tu = np.cross(n_unit, trial2)
            tu /= max(float(np.linalg.norm(tu)), 1e-12)
            tv = np.cross(n_unit, tu)
            ring_angles = np.linspace(0, 2 * np.pi, ring_n_pts)
            ring_pts = np.array([
                ct + ring_radius * (np.cos(a) * tu + np.sin(a) * tv)
                for a in ring_angles
            ])
            ax.plot(ring_pts[:, 0], ring_pts[:, 1], ring_pts[:, 2],
                    color="black", linewidth=1.2, alpha=0.95, zorder=24)

            # Big disc at the actual contact (always on top)
            ax.scatter(*label_pos, c=CONTACT_COLORS[i], s=700, alpha=0.20,
                       depthshade=False, zorder=27)
            ax.scatter(*label_pos, c=CONTACT_COLORS[i], s=380, alpha=0.40,
                       depthshade=False, zorder=28)
            ax.scatter(*label_pos, c=CONTACT_COLORS[i], s=180,
                       edgecolors="black", linewidths=2.0,
                       depthshade=False, zorder=30)
            # Bold label
            ax.text(label_pos[0], label_pos[1], label_pos[2],
                    f"  {CONTACT_LABELS[i]}",
                    color=CONTACT_COLORS[i], fontsize=16, weight="bold",
                    zorder=35,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])

            # Normal arrow at the actual contact
            arrow_len = 0.015
            ax.quiver(ct[0], ct[1], ct[2],
                      n[0] * arrow_len, n[1] * arrow_len, n[2] * arrow_len,
                      color=CONTACT_COLORS[i], arrow_length_ratio=0.3,
                      linewidth=1.5, alpha=0.8)

        # Approach: show the chosen grasp direction near the object.
        app_len = 0.04
        start = grasp_centre - approach * app_len
        ax.quiver(start[0], start[1], start[2],
                  approach[0] * app_len, approach[1] * app_len, approach[2] * app_len,
                  color="darkorange", arrow_length_ratio=0.25, linewidth=3.0)
        ax.text(*start, "  approach", color="darkorange",
                fontsize=11, weight="bold")

    # Left panel bounds: keep the overview centered on the OBJECT and its
    # grasp, not on the entire scene. The camera is still drawn as a
    # reference, but it no longer dictates the zoom level.
    overview_bounds = np.vstack([
        obj_vis_pts,
        np.array(contacts),
        np.array([grasp_centre]),
    ])
    overview_mins, overview_maxs = _set_padded_limits(ax, overview_bounds, pad_ratio=0.14, min_pad=0.01)
    if not PAPER_MODE:
        left_title = f"Overview — camera + {FRAME_PREFIX} + approach"
        if VIS_LABEL:
            left_title = f"{VIS_LABEL} — {left_title}"
        ax.set_title(left_title, fontsize=14)

    # Camera-frame anchor: keep the frame orientation visible near the plot
    # edge without pretending this is the true camera position.
    if not PAPER_MODE:
        overview_span = np.maximum(overview_maxs - overview_mins, 1e-6)
        cam_anchor = np.array([
            overview_maxs[0] - 0.10 * overview_span[0],
            overview_mins[1] + 0.10 * overview_span[1],
            overview_mins[2] + 0.06 * overview_span[2],
        ])

        axis_len = float(np.clip(np.median(obj_vis_pts.max(axis=0) - obj_vis_pts.min(axis=0)) * 0.35, 0.012, 0.025))
        ax.quiver(cam_anchor[0], cam_anchor[1], cam_anchor[2], axis_len, 0, 0, color="red",
                  linewidth=2, arrow_length_ratio=0.2)
        ax.quiver(cam_anchor[0], cam_anchor[1], cam_anchor[2], 0, axis_len, 0, color="green",
                  linewidth=2, arrow_length_ratio=0.2)
        ax.quiver(cam_anchor[0], cam_anchor[1], cam_anchor[2], 0, 0, axis_len, color="blue",
                  linewidth=2, arrow_length_ratio=0.2)
        ax.text(cam_anchor[0] + axis_len * 1.1, cam_anchor[1], cam_anchor[2], "X", color="red",
                fontsize=9, weight="bold")
        ax.text(cam_anchor[0], cam_anchor[1] + axis_len * 1.1, cam_anchor[2], "Y", color="green",
                fontsize=9, weight="bold")
        ax.text(cam_anchor[0], cam_anchor[1], cam_anchor[2] + axis_len * 1.1, "Z", color="blue",
                fontsize=9, weight="bold")
        ax.scatter(*cam_anchor, c="black", s=120, marker="o", zorder=15)
        ax.text(cam_anchor[0], cam_anchor[1], cam_anchor[2], "  camera frame", color="black",
                fontsize=9, weight="bold")

    # Default view: oblique perspective focused on the object.
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)

    if PAPER_MODE:
        ax.set_axis_off()
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(_m_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(_m_formatter))
        ax.zaxis.set_major_formatter(FuncFormatter(_m_formatter))
        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.set_zlabel("Z (m)", fontsize=10)
        ax.tick_params(labelsize=7)

    contact_arr = np.array(contacts)
    grasp_c = contact_arr.mean(axis=0)

    mode = best.get("gripper_mode", "?")
    rotation_deg = float(best.get("rotation_deg", 0.0))
    grip_type = best["scores"].get("grip_type", "?")
    if not PAPER_MODE:
        fig.suptitle(
            f"{FRAME_PREFIX} — "
            f"{VIS_LABEL + ' · ' if VIS_LABEL else ''}"
            f"patch {best['patch_id']}, "
            f"mode={mode}, rot={rotation_deg:.0f}deg, grip={grip_type}  ·  "
            f"score {best['scores']['final_score']:.4f}  "
            f"thickness={best['scores']['thickness_score']:.2f} "
            f"(Ø{best['scores'].get('local_diameter_cm', best['scores']['local_diameter_mm'] / 10.0):.2f}cm)  "
            f"triangle={best['scores']['triangle_score']:.2f}",
            fontsize=11, weight="bold", y=0.98,
        )

    if not PAPER_MODE:
        legend_handles = [
            Line2D([0], [0], color="red",   lw=2, label="X axis (camera frame)"),
            Line2D([0], [0], color="green", lw=2, label="Y axis (camera frame)"),
            Line2D([0], [0], color="blue",  lw=2, label="Z axis (camera frame)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
                   markersize=12, markeredgecolor="black",
                   label="Contact F1 (finger 1)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="lime",
                   markersize=12, markeredgecolor="black",
                   label="Contact F2 (finger 2)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="dodgerblue",
                   markersize=12, markeredgecolor="black",
                   label="Contact F3 (finger 3)"),
            Line2D([0], [0], color="black", lw=0.8, linestyle="--",
                   label="Contact triangle"),
        ]
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=8, frameon=True,
                  facecolor="white", edgecolor="black")

    # Enable mouse scroll-wheel zoom on BOTH 3D panels.
    # Drag with left mouse = rotate (default 3D behavior).
    # Drag with right mouse = pan (built-in in modern matplotlib).
    def _on_scroll(event):
        target = event.inaxes
        if target is None:
            return
        scale = 0.85 if event.button == "up" else 1.18
        for getter, setter in [
            (target.get_xlim, target.set_xlim),
            (target.get_ylim, target.set_ylim),
            (target.get_zlim, target.set_zlim),
        ]:
            lo, hi = getter()
            mid = (lo + hi) / 2
            half = (hi - lo) / 2 * scale
            setter(mid - half, mid + half)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", _on_scroll)

    if PAPER_MODE:
        plt.tight_layout()
    else:
        plt.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.90)

    out_png = OUTPUT_PNG
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"[save] {out_png}")

    if "agg" not in plt.get_backend().lower():
        try:
            plt.show()
        except Exception as exc:
            print(f"[warn] interactive display failed: {exc}")


if __name__ == "__main__":
    main()
