#!/usr/bin/env python3
"""
Interactive viewer for cup grasp — rotate to desired angle, then:
  - press  S  to save PDF with current view
  - close the window  to save PDF with current view

Run with:
  python3 interactive_cup_pdf.py baseline
  python3 interactive_cup_pdf.py heatmap
"""

from __future__ import annotations

import sys
from pathlib import Path

import json
import numpy as np

import matplotlib
# Force interactive backend — must be set before importing pyplot.
# 'macosx' is the native macOS backend (supports drag-to-rotate).
# Falls back to TkAgg if macosx is unavailable.
try:
    matplotlib.use("macosx")
except Exception:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ── Paths ────────────────────────────────────────────────────────────

GRASP_WORK = Path(__file__).resolve().parent
CASE_DIR = (
    GRASP_WORK
    / "refactored_sam_pipeline-3"
    / "outputs_capture_4_handcream_symmetric_v2_refacut"
    / "grasp_ready_objects"
    / "blue_cup"
    / "primary_symmetric_3d"
)

OBJECT_PLY     = CASE_DIR / "blue_cup_object_cloud.ply"
SCENE_PLY      = CASE_DIR / "blue_cup_scene_cloud.ply"
PATCH_DATA_NPZ = CASE_DIR / "blue_cup_grasp_patch_data.npz"
BASELINE_JSON  = GRASP_WORK / "grasp_scores_blue_cup_baseline.json"
HEATMAP_JSON   = GRASP_WORK / "grasp_scores_blue_cup_heatmap.json"

CONTACT_COLORS = ["red", "lime", "dodgerblue"]
CONTACT_LABELS = ["F1", "F2", "F3"]


# ── Helpers ──────────────────────────────────────────────────────────

def read_ply(path: Path) -> np.ndarray:
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


def project_heat(obj_pts: np.ndarray, patch_pts: np.ndarray, heat: np.ndarray) -> np.ndarray:
    result = np.zeros(len(obj_pts), dtype=np.float64)
    for i, op in enumerate(obj_pts):
        dists = np.linalg.norm(patch_pts - op, axis=1)
        result[i] = heat[int(np.argmin(dists))]
    return result


def robust_inlier_mask(points: np.ndarray, pct: float = 98.5) -> np.ndarray:
    if len(points) < 32:
        return np.ones(len(points), dtype=bool)
    ref = np.median(points, axis=0)
    d = np.linalg.norm(points - ref, axis=1)
    thresh = np.percentile(d, pct)
    mask = d <= max(float(thresh), 1e-6)
    if mask.sum() < max(16, int(0.35 * len(points))):
        return np.ones(len(points), dtype=bool)
    return mask


def set_limits(ax, points: np.ndarray, pad_ratio=0.14, min_pad=0.01):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, min_pad)
    pads = np.maximum(spans * pad_ratio, min_pad)
    ax.set_xlim(mins[0] - pads[0], maxs[0] + pads[0])
    ax.set_ylim(mins[1] - pads[1], maxs[1] + pads[1])
    ax.set_zlim(mins[2] - pads[2], maxs[2] + pads[2])
    ax.set_box_aspect(tuple((spans + 2.0 * pads).tolist()))


def draw_contacts(ax, contacts, normals):
    tri = np.array(contacts + [contacts[0]])
    ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], "k--",
            linewidth=1.0, alpha=0.5, zorder=15)

    for i, (ct, n) in enumerate(zip(contacts, normals)):
        n_unit = n / max(float(np.linalg.norm(n)), 1e-12)
        label_pos = ct + n_unit * 0.008

        ring_radius = 0.003
        trial2 = np.array([1.0, 0.0, 0.0])
        if abs(trial2 @ n_unit) > 0.9:
            trial2 = np.array([0.0, 1.0, 0.0])
        tu = np.cross(n_unit, trial2)
        tu /= max(float(np.linalg.norm(tu)), 1e-12)
        tv = np.cross(n_unit, tu)
        ring_angles = np.linspace(0, 2 * np.pi, 32)
        ring_pts = np.array([
            ct + ring_radius * (np.cos(a) * tu + np.sin(a) * tv)
            for a in ring_angles
        ])
        ax.plot(ring_pts[:, 0], ring_pts[:, 1], ring_pts[:, 2],
                color="black", linewidth=1.2, alpha=0.95, zorder=24)

        ax.scatter(*label_pos, c=CONTACT_COLORS[i], s=700, alpha=0.20, depthshade=False, zorder=27)
        ax.scatter(*label_pos, c=CONTACT_COLORS[i], s=380, alpha=0.40, depthshade=False, zorder=28)
        ax.scatter(*label_pos, c=CONTACT_COLORS[i], s=180,
                   edgecolors="black", linewidths=2.0, depthshade=False, zorder=30)
        ax.text(label_pos[0], label_pos[1], label_pos[2],
                f"  {CONTACT_LABELS[i]}",
                color=CONTACT_COLORS[i], fontsize=16, weight="bold", zorder=35,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])

        arrow_len = 0.015
        ax.quiver(ct[0], ct[1], ct[2],
                  n[0] * arrow_len, n[1] * arrow_len, n[2] * arrow_len,
                  color=CONTACT_COLORS[i], arrow_length_ratio=0.3, linewidth=1.5, alpha=0.8)


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "baseline"
    if mode not in ("baseline", "heatmap"):
        print("Usage: python3 interactive_cup_pdf.py [baseline|heatmap]")
        sys.exit(1)

    use_heat = (mode == "heatmap")
    out_pdf = GRASP_WORK / f"cup_{mode}.pdf"
    scores_json = HEATMAP_JSON if use_heat else BASELINE_JSON

    # Load data
    obj_pts   = read_ply(OBJECT_PLY)
    scene_pts = read_ply(SCENE_PLY) if SCENE_PLY.exists() else np.zeros((0, 3))
    npz       = np.load(PATCH_DATA_NPZ, allow_pickle=True)
    patch_pts = np.asarray(npz["points"], dtype=np.float64)
    raw_heat  = np.clip(np.asarray(npz["point_heat_values"], dtype=np.float64), 0.0, 1.0)

    best     = json.loads(scores_json.read_text())["best_candidate"]
    contacts = [np.array(c["point_xyz_m"]) for c in best["contacts"]]
    normals  = [np.array(c["normal_xyz_unit"]) for c in best["contacts"]]
    approach = np.array(best["approach_direction_xyz_unit"])

    inlier_mask = robust_inlier_mask(obj_pts)
    vis_pts     = obj_pts[inlier_mask]
    obj_step    = max(1, len(vis_pts) // 5000)

    obj_heat = None
    if use_heat:
        print("Projecting heat values ...")
        obj_heat = project_heat(obj_pts, patch_pts, raw_heat)

    # Build figure
    fig = plt.figure(figsize=(8, 7.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # Scene context
    if len(scene_pts) > 0:
        obj_center = vis_pts.mean(axis=0)
        obj_extent = vis_pts.max(axis=0) - vis_pts.min(axis=0)
        radius = max(float(np.linalg.norm(obj_extent)) * 1.25, 0.12)
        local_mask = np.linalg.norm(scene_pts - obj_center, axis=1) <= radius
        local_scene = scene_pts[local_mask] if local_mask.any() else scene_pts
        s_step = max(1, len(local_scene) // 6000)
        s = local_scene[::s_step]
        ax.scatter(s[:, 0], s[:, 1], s[:, 2],
                   c="lightgrey", s=0.4, alpha=0.18, depthshade=True)

    # Object cloud
    o = vis_pts[::obj_step]
    if use_heat and obj_heat is not None:
        hv = obj_heat[inlier_mask][::obj_step]
        ax.scatter(o[:, 0], o[:, 1], o[:, 2],
                   c=hv, cmap="jet", s=2.0, alpha=0.85,
                   vmin=0.0, vmax=1.0, depthshade=False)
    else:
        ax.scatter(o[:, 0], o[:, 1], o[:, 2],
                   c="steelblue", s=1.2, alpha=0.7)

    draw_contacts(ax, contacts, normals)

    # Approach arrow
    grasp_centre = np.mean(contacts, axis=0)
    app_len = 0.04
    start = grasp_centre - approach * app_len
    ax.quiver(start[0], start[1], start[2],
              approach[0] * app_len, approach[1] * app_len, approach[2] * app_len,
              color="darkorange", arrow_length_ratio=0.25, linewidth=3.0)
    ax.text(*start, "  approach", color="darkorange", fontsize=11, weight="bold")

    bounds = np.vstack([vis_pts, np.array(contacts), np.array([grasp_centre])])
    set_limits(ax, bounds)

    ax.view_init(elev=18, azim=-35)
    ax.set_axis_off()
    plt.tight_layout()

    fig.suptitle(
        f"Cup — {mode}  |  roteste si apasa  S  sau inchide fereastra pentru a salva PDF",
        fontsize=10, color="gray", y=0.99,
    )

    saved = [False]

    def save_pdf():
        if saved[0]:
            return
        saved[0] = True
        elev = ax.elev
        azim = ax.azim
        fig.suptitle("")   # sterge titlul instructional
        plt.tight_layout()
        fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
        print(f"\nSaved: {out_pdf}  (elev={elev:.1f}, azim={azim:.1f})")

    def on_key(event):
        if event.key and event.key.lower() == "s":
            save_pdf()

    def on_close(_event):
        save_pdf()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    print(f"\nMode: {mode}")
    print("Roteste cum vrei, apoi apasa  S  sau inchide fereastra pentru a salva PDF.")
    plt.show()


if __name__ == "__main__":
    main()
