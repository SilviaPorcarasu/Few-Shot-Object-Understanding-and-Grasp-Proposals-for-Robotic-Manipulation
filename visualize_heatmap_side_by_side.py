#!/usr/bin/env python3
"""
Interactive side-by-side grasp comparison:

  LEFT  = full reconstruction, no heatmap prior
  RIGHT = same reconstruction, with heatmap prior

Both panels stay rotatable / zoomable, and their camera angle is synced after
you rotate one of them so the visual comparison stays easy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from grasp_config import FRAME_DIR as DEFAULT_FRAME_DIR, HEATMAP_DIR as DEFAULT_HEATMAP_DIR
from grasp_helpers import project_sparse_point_heat_to_object
from run_grasps_on_sam3_outputs import (
    DATA_DIR,
    build_roll_symmetric_object_cloud,
    build_semantic_cylinder_object_cloud,
    canonical_source_prefix,
    iter_available_output_dirs,
    stem_to_prefix,
)
from visualize_grasp import (
    CONTACT_COLORS,
    CONTACT_LABELS,
    _m_formatter,
    _robust_inlier_mask,
    _set_padded_limits,
    read_ply,
)

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent


def _infer_prefix(case_dir: Path) -> str:
    object_cloud = next(case_dir.glob("*_object_cloud.ply"), None)
    if object_cloud is None:
        raise RuntimeError(f"Could not infer prefix from {case_dir}")
    return object_cloud.name.replace("_object_cloud.ply", "")


def _load_completed_display_cloud(
    case_dir: Path,
    prefix: str,
    fallback_points: np.ndarray,
) -> np.ndarray:
    """
    Prefer a denser completed display cloud for symmetric semantic objects.

    This keeps grasping itself unchanged, but makes the no-heatmap panel show
    the full completed surface implied by the upstream SAM3 semantic model.
    """
    dense_case_cloud = case_dir / f"{prefix}_object_cloud_no_heatmap_dense.ply"
    if dense_case_cloud.exists():
        return read_ply(dense_case_cloud)

    if case_dir.name != "primary_symmetric_3d":
        return fallback_points

    for outputs_dir in iter_available_output_dirs():
        sym_root = outputs_dir / "symmetric_semantic_3d"
        if not sym_root.exists():
            continue
        for obj_dir in sorted(p for p in sym_root.iterdir() if p.is_dir()):
            if canonical_source_prefix(stem_to_prefix(obj_dir.name)) != prefix:
                continue
            meta_path = obj_dir / "metadata.json"
            if not meta_path.exists():
                continue
            metadata = json.loads(meta_path.read_text())
            if metadata.get("roll_model") and not metadata.get("semantic_cylinder"):
                return build_roll_symmetric_object_cloud(obj_dir)
            if metadata.get("semantic_cylinder"):
                return build_semantic_cylinder_object_cloud(obj_dir, metadata)
            combined_object = obj_dir / "combined_object_original_symmetric.ply"
            object_cloud = combined_object if combined_object.exists() else obj_dir / "object_cloud.ply"
            if object_cloud.exists():
                return read_ply(object_cloud)
    return fallback_points


def _resolve_scores_path(case_dir: Path, filename: str) -> Path:
    local = case_dir / filename
    root = PROJECT_ROOT / filename
    if local.exists() and root.exists():
        return root if root.stat().st_mtime >= local.stat().st_mtime else local
    if root.exists():
        return root
    return local


def _load_dense_visible_mask_cloud(obj_dir: Path, metadata: dict) -> np.ndarray:
    """Backproject the union of stored patch-mask pixels into a dense visible surface."""
    image_ref = metadata.get("image_path", "")
    image_name = Path(image_ref).name
    frame_id = image_name.replace("rgb_", "").replace(".png", "")
    depth_path = DATA_DIR / f"depth_{frame_id}.npy"
    if not depth_path.exists():
        return np.zeros((0, 3), dtype=np.float64)

    pixels = []
    for npz_path in sorted(obj_dir.glob("patch_*.npz")):
        npz = np.load(npz_path, allow_pickle=True)
        if "original_pixels_uv" in npz.files:
            pixels.append(np.asarray(npz["original_pixels_uv"], dtype=np.int32))
    if not pixels:
        return np.zeros((0, 3), dtype=np.float64)

    uv = np.unique(np.concatenate(pixels, axis=0), axis=0)
    depth_m = np.asarray(np.load(depth_path), dtype=np.float32)
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]
    intr = metadata.get("intrinsics", {})
    fx = float(intr.get("fx", 0.0))
    fy = float(intr.get("fy", 0.0))
    cx = float(intr.get("cx", 0.0))
    cy = float(intr.get("cy", 0.0))
    if min(fx, fy) <= 0.0:
        return np.zeros((0, 3), dtype=np.float64)
    h, w = depth_m.shape[:2]
    u = np.clip(uv[:, 0], 0, w - 1)
    v = np.clip(uv[:, 1], 0, h - 1)
    z = depth_m[v, u].astype(np.float64)
    valid = np.isfinite(z) & (z > 0.0) & (z <= 3.0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)
    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    z = z[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.column_stack((x, y, z))


def _load_case_data(
    case_dir: Path,
    prefix: str,
    scores_path: Path,
    *,
    use_heatmap: bool,
    heatmap_dir: Path,
) -> dict:
    npz_path = case_dir / f"{prefix}_grasp_patch_data.npz"
    object_ply_path = case_dir / f"{prefix}_object_cloud.ply"
    scene_ply_path = case_dir / f"{prefix}_scene_cloud.ply"

    npz = np.load(npz_path, allow_pickle=True)
    patch_pts = np.asarray(npz["points"], dtype=np.float64)
    patch_pixels = np.asarray(npz["pixels"], dtype=np.int32)
    frame_name = str(npz["frame_name"])
    point_heat_values = (
        np.asarray(npz["point_heat_values"], dtype=np.float64)
        if "point_heat_values" in npz.files
        else None
    )

    obj_pts = read_ply(object_ply_path)
    scene_pts = read_ply(scene_ply_path) if scene_ply_path.exists() else np.zeros((0, 3), dtype=np.float64)
    report = json.loads(scores_path.read_text())
    best = report.get("best_candidate")
    camera_origin = np.asarray(
        report.get("camera_origin_xyz_m", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    object_center = np.asarray(
        report.get("object_center_xyz_m", obj_pts.mean(axis=0).tolist()),
        dtype=np.float64,
    )

    obj_inlier_mask = _robust_inlier_mask(obj_pts)
    obj_vis_idx = np.flatnonzero(obj_inlier_mask)
    if len(obj_vis_idx) == 0:
        obj_vis_idx = np.arange(len(obj_pts))
    obj_vis_pts = obj_pts[obj_vis_idx]

    use_point_heat_display = (
        use_heatmap
        and point_heat_values is not None
        and len(point_heat_values) == len(patch_pts)
    )
    obj_hm_values = None
    if use_point_heat_display:
        obj_hm_values = project_sparse_point_heat_to_object(
            obj_pts,
            patch_pts,
            np.clip(point_heat_values, 0.0, 1.0),
        )

    return {
        "prefix": prefix,
        "case_dir": case_dir,
        "report": report,
        "best": best,
        "use_heatmap": use_heatmap,
        "use_point_heat_display": use_point_heat_display,
        "obj_pts": obj_pts,
        "obj_vis_idx": obj_vis_idx,
        "obj_vis_pts": obj_vis_pts,
        "obj_hm_values": obj_hm_values,
        "camera_origin": camera_origin,
        "object_center": object_center,
        "patch_pts": patch_pts,
        "patch_pixels": patch_pixels,
        "patch_heat_values": point_heat_values,
        "scene_pts": scene_pts,
        "frame_name": frame_name,
        "heatmap_dir": heatmap_dir,
    }


def _plot_camera(ax, camera_origin: np.ndarray, axis_len: float, label: str = "camera (0,0,0)") -> None:
    ax.scatter(*camera_origin, c="black", s=90, zorder=20)
    ax.quiver(*camera_origin, axis_len, 0, 0, color="red", linewidth=2, arrow_length_ratio=0.2)
    ax.quiver(*camera_origin, 0, axis_len, 0, color="green", linewidth=2, arrow_length_ratio=0.2)
    ax.quiver(*camera_origin, 0, 0, axis_len, color="blue", linewidth=2, arrow_length_ratio=0.2)
    ax.text(
        camera_origin[0],
        camera_origin[1],
        camera_origin[2],
        f"  {label}",
        fontsize=9,
        weight="bold",
        color="black",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )


def _draw_placeholder(ax, prefix: str, panel_label: str) -> None:
    ax.set_axis_off()
    ax.text2D(
        0.05,
        0.62,
        f"{prefix} — {panel_label}",
        transform=ax.transAxes,
        fontsize=14,
        weight="bold",
    )
    ax.text2D(
        0.05,
        0.48,
        "No grasp candidate was generated for this case.",
        transform=ax.transAxes,
        fontsize=11,
        color="dimgray",
    )


def _plot_camera_direction_marker(
    ax,
    camera_origin: np.ndarray,
    object_center: np.ndarray,
    object_span: np.ndarray,
) -> None:
    cam_vec = camera_origin - object_center
    cam_dist_m = float(np.linalg.norm(cam_vec))
    if cam_dist_m <= 1e-12:
        return
    cam_dir = cam_vec / cam_dist_m
    display_dist = float(np.clip(np.median(object_span) * 0.75, 0.035, 0.08))
    display_pos = object_center + cam_dir * display_dist
    axis_len = float(np.clip(np.median(object_span) * 0.22, 0.012, 0.022))

    ax.plot(
        [object_center[0], display_pos[0]],
        [object_center[1], display_pos[1]],
        [object_center[2], display_pos[2]],
        linestyle="--",
        color="gray",
        linewidth=1.4,
        alpha=0.9,
    )
    _plot_camera(ax, display_pos, axis_len=axis_len, label=f"camera ({cam_dist_m * 100.0:.1f} cm)")


def _plot_case(ax, data: dict, *, panel_label: str, show_legend: bool) -> None:
    best = data["best"]
    prefix = data["prefix"]
    if best is None:
        _draw_placeholder(ax, prefix, panel_label)
        return

    obj_pts = data["obj_pts"]
    obj_vis_idx = data["obj_vis_idx"]
    obj_vis_pts = data["obj_vis_pts"]
    obj_hm_values = data["obj_hm_values"]
    patch_pts = data["patch_pts"]
    patch_heat_values = data["patch_heat_values"]
    scene_pts = data["scene_pts"]
    use_heatmap = bool(data["use_heatmap"])
    use_point_heat_display = bool(data["use_point_heat_display"])
    camera_origin = data["camera_origin"]
    object_center = data["object_center"]

    contacts = [np.array(c["point_xyz_m"], dtype=np.float64) for c in best["contacts"]]
    normals = [np.array(c["normal_xyz_unit"], dtype=np.float64) for c in best["contacts"]]
    approach = np.array(best["approach_direction_xyz_unit"], dtype=np.float64)
    grasp_centre = np.mean(contacts, axis=0)

    o_idx = obj_vis_idx[:: max(1, len(obj_vis_idx) // 5000)]
    o = obj_pts[o_idx]
    if use_heatmap and patch_heat_values is not None and len(patch_heat_values) == len(patch_pts):
        ax.scatter(
            obj_vis_pts[:, 0], obj_vis_pts[:, 1], obj_vis_pts[:, 2],
            c="steelblue", s=0.9, alpha=0.16, depthshade=False,
        )
        ax.scatter(
            o[:, 0], o[:, 1], o[:, 2],
            c="steelblue", s=1.2, alpha=0.34, depthshade=False,
        )
        p_idx = np.arange(0, len(patch_pts), max(1, len(patch_pts) // 5000))
        p = patch_pts[p_idx]
        hm_vals = np.clip(patch_heat_values[p_idx], 0.0, 1.0)
        ax.scatter(
            p[:, 0], p[:, 1], p[:, 2],
            c=hm_vals,
            cmap="jet",
            s=2.2,
            alpha=0.90,
            vmin=0.0,
            vmax=1.0,
            depthshade=False,
        )
    else:
        ax.scatter(
            obj_vis_pts[:, 0], obj_vis_pts[:, 1], obj_vis_pts[:, 2],
            c="steelblue", s=0.9, alpha=0.16, depthshade=False,
        )
        ax.scatter(
            o[:, 0], o[:, 1], o[:, 2],
            c="steelblue", s=1.2, alpha=0.34, depthshade=False,
        )

    tri = np.array(contacts + [contacts[0]])
    ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], "k--", linewidth=1.0, alpha=0.5, zorder=15)

    pregrasp_standoff_m = 0.008
    for i, (ct, n) in enumerate(zip(contacts, normals)):
        n_unit = n / max(float(np.linalg.norm(n)), 1e-12)
        label_pos = ct + n_unit * pregrasp_standoff_m

        trial = np.array([1.0, 0.0, 0.0])
        if abs(float(trial @ n_unit)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        tu = np.cross(n_unit, trial)
        tu /= max(float(np.linalg.norm(tu)), 1e-12)
        tv = np.cross(n_unit, tu)
        ring_angles = np.linspace(0.0, 2.0 * np.pi, 32)
        ring_pts = np.array([
            ct + 0.003 * (np.cos(a) * tu + np.sin(a) * tv)
            for a in ring_angles
        ])
        ax.plot(ring_pts[:, 0], ring_pts[:, 1], ring_pts[:, 2], color="black", linewidth=1.2, alpha=0.95, zorder=24)

        ax.scatter(*label_pos, c=CONTACT_COLORS[i], s=700, alpha=0.20, depthshade=False, zorder=27)
        ax.scatter(*label_pos, c=CONTACT_COLORS[i], s=380, alpha=0.40, depthshade=False, zorder=28)
        ax.scatter(
            *label_pos,
            c=CONTACT_COLORS[i],
            s=180,
            edgecolors="black",
            linewidths=2.0,
            depthshade=False,
            zorder=30,
        )
        ax.text(
            label_pos[0],
            label_pos[1],
            label_pos[2],
            f"  {CONTACT_LABELS[i]}",
            color=CONTACT_COLORS[i],
            fontsize=16,
            weight="bold",
            zorder=35,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        )

        arrow_len = 0.015
        ax.quiver(
            ct[0], ct[1], ct[2],
            n[0] * arrow_len, n[1] * arrow_len, n[2] * arrow_len,
            color=CONTACT_COLORS[i],
            arrow_length_ratio=0.3,
            linewidth=1.5,
            alpha=0.8,
        )

    app_len = 0.04
    start = grasp_centre - approach * app_len
    ax.quiver(
        start[0], start[1], start[2],
        approach[0] * app_len, approach[1] * app_len, approach[2] * app_len,
        color="darkorange",
        arrow_length_ratio=0.25,
        linewidth=3.0,
    )
    ax.text(*start, "  approach", color="darkorange", fontsize=11, weight="bold")

    overview_bounds = np.vstack([
        obj_vis_pts,
        np.array(contacts),
        np.array([grasp_centre]),
    ])
    overview_mins, overview_maxs = _set_padded_limits(ax, overview_bounds, pad_ratio=0.14, min_pad=0.01)
    ax.set_title(f"{panel_label} — {prefix}", fontsize=14)

    object_span = np.maximum(obj_vis_pts.max(axis=0) - obj_vis_pts.min(axis=0), 1e-6)
    _plot_camera_direction_marker(ax, camera_origin, object_center, object_span)

    ax.view_init(elev=18, azim=-35)
    ax.xaxis.set_major_formatter(FuncFormatter(_m_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(_m_formatter))
    ax.zaxis.set_major_formatter(FuncFormatter(_m_formatter))
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_zlabel("Z (m)", fontsize=10)
    ax.tick_params(labelsize=7)

    if show_legend:
        legend_handles = [
            Line2D([0], [0], color="red", lw=2, label="X axis (camera frame)"),
            Line2D([0], [0], color="green", lw=2, label="Y axis (camera frame)"),
            Line2D([0], [0], color="blue", lw=2, label="Z axis (camera frame)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=12, markeredgecolor="black", label="Contact F1"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="lime", markersize=12, markeredgecolor="black", label="Contact F2"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="dodgerblue", markersize=12, markeredgecolor="black", label="Contact F3"),
            Line2D([0], [0], color="black", lw=0.8, linestyle="--", label="Contact triangle"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8,
            frameon=True,
            facecolor="white",
            edgecolor="black",
        )


def _attach_scroll_zoom(fig) -> None:
    def _on_scroll(event):
        target = event.inaxes
        if target is None:
            return
        scale = 0.85 if event.button == "up" else 1.18
        for ax in fig.axes:
            if not hasattr(ax, "get_xlim"):
                continue
            for getter, setter in [
                (ax.get_xlim, ax.set_xlim),
                (ax.get_ylim, ax.set_ylim),
                (ax.get_zlim, ax.set_zlim),
            ]:
                lo, hi = getter()
                mid = (lo + hi) / 2.0
                half = (hi - lo) / 2.0 * scale
                setter(mid - half, mid + half)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", _on_scroll)


def _attach_view_sync(fig, axes: list) -> None:
    state = {"syncing": False, "last": {}}

    def _snapshot(ax) -> tuple:
        return (
            round(float(ax.elev), 6),
            round(float(ax.azim), 6),
            round(float(getattr(ax, "roll", 0.0)), 6),
            tuple(round(float(v), 6) for v in ax.get_xlim3d()),
            tuple(round(float(v), 6) for v in ax.get_ylim3d()),
            tuple(round(float(v), 6) for v in ax.get_zlim3d()),
        )

    def _copy_view(source) -> None:
        if state["syncing"]:
            return
        state["syncing"] = True
        try:
            elev = float(source.elev)
            azim = float(source.azim)
            roll = float(getattr(source, "roll", 0.0))
            xlim = source.get_xlim3d()
            ylim = source.get_ylim3d()
            zlim = source.get_zlim3d()
            for ax in axes:
                if ax is source:
                    continue
                ax.view_init(elev=elev, azim=azim, roll=roll)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)
            for ax in axes:
                state["last"][ax] = _snapshot(ax)
            fig.canvas.draw_idle()
        finally:
            state["syncing"] = False

    def _maybe_sync(_event):
        if state["syncing"]:
            return
        for ax in axes:
            current = _snapshot(ax)
            previous = state["last"].get(ax)
            if previous is None:
                state["last"][ax] = current
                continue
            if current != previous:
                _copy_view(ax)
                break

    for ax in axes:
        state["last"][ax] = _snapshot(ax)

    fig.canvas.mpl_connect("motion_notify_event", _maybe_sync)
    fig.canvas.mpl_connect("button_release_event", _maybe_sync)


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive side-by-side no-heatmap vs heatmap grasp viewer.")
    parser.add_argument("--case-dir", default=str(DEFAULT_FRAME_DIR), help="Grasp-ready case directory.")
    parser.add_argument("--prefix", help="Object prefix. Inferred from *_object_cloud.ply when omitted.")
    parser.add_argument("--output", help="Optional PNG output path.")
    args = parser.parse_args()

    case_dir = Path(args.case_dir).expanduser()
    prefix = args.prefix or _infer_prefix(case_dir)
    heatmap_dir = DEFAULT_HEATMAP_DIR
    no_scores = _resolve_scores_path(case_dir, "grasp_scores_no_heatmap.json")
    with_scores = _resolve_scores_path(case_dir, "grasp_scores_with_heatmap.json")
    if not no_scores.exists() or not with_scores.exists():
        raise RuntimeError(
            "Missing comparison score files. Expected both "
            f"{no_scores.name} and {with_scores.name} in {case_dir}"
        )

    no_data = _load_case_data(case_dir, prefix, no_scores, use_heatmap=False, heatmap_dir=heatmap_dir)
    with_data = _load_case_data(case_dir, prefix, with_scores, use_heatmap=True, heatmap_dir=heatmap_dir)

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor("white")
    ax_left = fig.add_subplot(121, projection="3d")
    ax_right = fig.add_subplot(122, projection="3d")
    ax_left.set_facecolor("white")
    ax_right.set_facecolor("white")

    _plot_case(ax_left, no_data, panel_label="No heatmap", show_legend=False)
    _plot_case(ax_right, with_data, panel_label="With heatmap", show_legend=True)

    shared_bounds = np.vstack([
        no_data["obj_vis_pts"],
        with_data["obj_vis_pts"],
    ])
    _set_padded_limits(ax_left, shared_bounds, pad_ratio=0.14, min_pad=0.01)
    _set_padded_limits(ax_right, shared_bounds, pad_ratio=0.14, min_pad=0.01)
    ax_left.view_init(elev=18, azim=-35)
    ax_right.view_init(elev=18, azim=-35)

    left_best = no_data["best"]
    right_best = with_data["best"]
    left_score = left_best["scores"]["final_score"] if left_best else None
    right_score = right_best["scores"]["final_score"] if right_best else None
    fig.suptitle(
        f"{prefix} — raw object-cloud comparison  ·  "
        f"left score={left_score:.4f} / right score={right_score:.4f}"
        if left_score is not None and right_score is not None
        else f"{prefix} — raw object-cloud comparison",
        fontsize=12,
        weight="bold",
        y=0.97,
    )

    _attach_scroll_zoom(fig)
    _attach_view_sync(fig, [ax_left, ax_right])
    plt.tight_layout()

    output_png = (
        Path(args.output).expanduser()
        if args.output
        else case_dir / f"grasp_interactive_compare_{prefix}_{case_dir.name}.png"
    )
    output_pdf = output_png.with_suffix(".pdf")
    fig.savefig(output_png, dpi=140, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=140, bbox_inches="tight")
    print(f"[save] {output_png}")
    print(f"[save] {output_pdf}")

    if "agg" not in plt.get_backend().lower():
        try:
            plt.show()
        except Exception as exc:
            print(f"[warn] interactive display failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
