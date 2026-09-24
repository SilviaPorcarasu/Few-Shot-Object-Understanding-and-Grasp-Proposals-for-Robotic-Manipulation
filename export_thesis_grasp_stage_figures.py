#!/usr/bin/env python3
"""
Export 3 thesis-ready figures that explain the grasping pipeline stages
on one point cloud case (default: blue_cup / primary_symmetric_3d).

Saved figures:
  1. stage1_patch_selection.png
  2. stage2_triangle_generation.png
  3. stage3_candidate_ranking.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
import numpy as np

import run_grasps
from compute_frames import compute_frames
from generate_contacts import generate_contacts
from grasp_helpers import project_sparse_point_heat_to_object, read_open3d_binary_ply
from object_dimensions import get_object_dimensions
from score_grasps import score_and_rank
from select_patches import select_patches


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_CASE_DIR = (
    PROJECT_ROOT
    / "refactored_sam_pipeline-2"
    / "grasp_ready_objects"
    / "blue_cup"
    / "primary_symmetric_3d"
)

CONTACT_COLORS = ["#ff3b30", "#32d74b", "#0a84ff"]
CONTACT_LABELS = ["F1", "F2", "F3"]


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v.copy()
    return v / n


def _set_limits(ax, points: np.ndarray, pad_ratio: float = 0.12, min_pad: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = np.maximum(span * pad_ratio, min_pad)
    mins -= pad
    maxs += pad
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(np.maximum(maxs - mins, 1e-6))
    return mins, maxs


def _style_3d(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_zlabel("Z (m)", fontsize=10)


def _camera_distance_text(camera_origin: np.ndarray, object_center: np.ndarray) -> str:
    dist = float(np.linalg.norm(object_center - camera_origin))
    return f"camera-object distance = {dist:.3f} m"


def _load_case(case_dir: Path) -> dict:
    prefix = case_dir.name.replace("primary_", "").replace("comparison_", "")
    # Real file prefix stays the object name, not the folder suffix.
    npz_path = next(case_dir.glob("*_grasp_patch_data.npz"))
    object_ply = next(case_dir.glob("*_object_cloud.ply"))
    patch_ply = next(case_dir.glob("*_grasp_patch_cloud.ply"))
    scores_path = case_dir / "grasp_scores.json"

    prefix = object_ply.name.replace("_object_cloud.ply", "")

    npz = np.load(npz_path, allow_pickle=True)
    camera_origin = np.asarray(npz["camera_origin"], dtype=np.float64)
    patch_heat_values = (
        np.asarray(npz["point_heat_values"], dtype=np.float64)
        if "point_heat_values" in npz.files
        else None
    )

    object_points, object_normals = read_open3d_binary_ply(object_ply)
    patch_points, _patch_normals = read_open3d_binary_ply(patch_ply)

    fx, fy, cx, cy = run_grasps._load_intrinsics()
    obj_pixels = run_grasps._project_3d_to_pixel(
        object_points, fx, fy, cx, cy,
    ).astype(np.int32)

    object_center = object_points.mean(axis=0)
    bbox_extent = object_points.max(axis=0) - object_points.min(axis=0)
    object_scale_m = float(np.linalg.norm(bbox_extent))

    centred = object_points - object_center
    cov = (centred.T @ centred) / max(len(centred) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    object_principal_axis = _normalize(eigvecs[:, 0])
    lam1 = float(max(eigvals[0], 1e-12))
    lam2 = float(max(eigvals[1], 0.0))
    object_elongation = float(np.clip(1.0 - (lam2 / lam1), 0.0, 1.0))

    if patch_heat_values is not None and len(patch_heat_values) == len(patch_points):
        grasp_heat = project_sparse_point_heat_to_object(
            object_points, patch_points, np.clip(patch_heat_values, 0.0, 1.0),
        )
    else:
        grasp_heat = None

    grasp_points = object_points
    grasp_pixels = obj_pixels
    grasp_normals = object_normals
    grasp_hm_values = grasp_heat

    object_prior = get_object_dimensions(prefix)
    if object_prior is not None and object_prior.canonical_name == "cup":
        (
            grasp_points,
            grasp_pixels,
            grasp_normals,
            grasp_hm_values,
            exterior_stats,
        ) = run_grasps._filter_cup_exterior_search_space(
            grasp_points,
            grasp_pixels,
            grasp_normals,
            grasp_hm_values,
            object_center,
            object_principal_axis,
        )
    else:
        exterior_stats = None

    patches = select_patches(
        grasp_points,
        grasp_hm_values,
        grasp_pixels=grasp_pixels,
    )
    frames = compute_frames(patches, grasp_points, camera_origin)
    candidates = generate_contacts(
        frames,
        grasp_points,
        grasp_pixels,
        grasp_normals,
        camera_origin,
        None,
        object_principal_axis,
        point_heat_values=grasp_hm_values,
    )
    scored = score_and_rank(
        candidates,
        object_center,
        object_scale_m,
        object_points,
        object_points,
        object_principal_axis,
        object_elongation,
        object_prior,
    )

    scores_json = json.loads(scores_path.read_text()) if scores_path.exists() else None

    return {
        "case_dir": case_dir,
        "prefix": prefix,
        "camera_origin": camera_origin,
        "object_points": object_points,
        "object_normals": object_normals,
        "object_center": object_center,
        "object_principal_axis": object_principal_axis,
        "grasp_points": grasp_points,
        "grasp_pixels": grasp_pixels,
        "grasp_normals": grasp_normals,
        "grasp_hm_values": grasp_hm_values,
        "patches": patches,
        "frames": frames,
        "candidates": candidates,
        "scored": scored,
        "scores_json": scores_json,
        "exterior_stats": exterior_stats,
    }


def _match_best_scored(case: dict):
    target = None
    if case["scores_json"] is not None:
        target = case["scores_json"].get("best_candidate")
    if target is None:
        return case["scored"][0]

    target_patch = int(target["patch_id"])
    target_mode = str(target["gripper_mode"])
    target_rot = float(target.get("rotation_deg", 0.0))
    target_spread_mm = float(target.get("finger_spread_mm", 0.0))

    for sg in case["scored"]:
        c = sg.candidate
        spread_mm = round(float(c.finger_spread_m) * 1000.0, 2)
        if (
            c.frame.patch.patch_id == target_patch
            and c.gripper_mode == target_mode
            and abs(float(c.rotation_deg) - target_rot) < 1e-6
            and abs(spread_mm - target_spread_mm) < 1e-3
        ):
            return sg

    return case["scored"][0]


def _plot_camera(ax, camera_origin: np.ndarray, axis_len: float = 0.03, label: str = "camera (0,0,0)") -> None:
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


def _plot_camera_anchor(ax, bounds_points: np.ndarray, label: str = "camera frame") -> None:
    mins = np.min(bounds_points, axis=0)
    maxs = np.max(bounds_points, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    anchor = np.array([
        maxs[0] - 0.10 * span[0],
        mins[1] + 0.10 * span[1],
        mins[2] + 0.06 * span[2],
    ])
    axis_len = float(np.clip(np.median(span) * 0.22, 0.012, 0.025))
    ax.scatter(*anchor, c="black", s=80, zorder=25)
    ax.quiver(*anchor, axis_len, 0, 0, color="red", linewidth=2, arrow_length_ratio=0.2)
    ax.quiver(*anchor, 0, axis_len, 0, color="green", linewidth=2, arrow_length_ratio=0.2)
    ax.quiver(*anchor, 0, 0, axis_len, color="blue", linewidth=2, arrow_length_ratio=0.2)
    ax.text(
        anchor[0], anchor[1], anchor[2],
        f"  {label}",
        fontsize=9, weight="bold", color="black",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )


def _plot_contacts(ax, contacts: list[np.ndarray], normals: list[np.ndarray], labels: list[str]) -> None:
    for idx, (pt, n, label) in enumerate(zip(contacts, normals, labels)):
        color = CONTACT_COLORS[idx]
        ax.scatter(*pt, c=color, s=380, edgecolors="black", linewidths=2, zorder=30)
        ax.text(
            pt[0], pt[1], pt[2], f"  {label}",
            color=color, fontsize=18, weight="bold", zorder=31,
            path_effects=[pe.withStroke(linewidth=4, foreground="white")],
        )
        n_unit = _normalize(n)
        ax.quiver(
            pt[0], pt[1], pt[2],
            n_unit[0] * 0.018, n_unit[1] * 0.018, n_unit[2] * 0.018,
            color=color, linewidth=2, arrow_length_ratio=0.25,
        )


def _export_stage1(case: dict, chosen_sg, out_path: Path) -> None:
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    for ax in (ax1, ax2):
        _style_3d(ax)

    object_points = case["object_points"]
    grasp_points = case["grasp_points"]
    camera_origin = case["camera_origin"]
    object_center = case["object_center"]
    chosen_patch = chosen_sg.candidate.frame.patch
    selected_patch_pts = grasp_points[chosen_patch.point_indices]
    seed_points = np.array([p.seed_point for p in case["patches"]], dtype=np.float64)

    ax1.scatter(
        object_points[:, 0], object_points[:, 1], object_points[:, 2],
        c="#c8c8c8", s=7, alpha=0.20, depthshade=False,
    )
    ax1.scatter(
        selected_patch_pts[:, 0], selected_patch_pts[:, 1], selected_patch_pts[:, 2],
        c="#ff8c00", s=12, alpha=0.95, depthshade=False,
    )
    ax1.scatter(seed_points[:, 0], seed_points[:, 1], seed_points[:, 2], c="black", s=16, alpha=0.8)
    _plot_camera(ax1, camera_origin)
    ax1.plot(
        [camera_origin[0], object_center[0]],
        [camera_origin[1], object_center[1]],
        [camera_origin[2], object_center[2]],
        linestyle="--",
        color="gray",
        linewidth=1.8,
    )
    mid = 0.5 * (camera_origin + object_center)
    ax1.text(
        mid[0], mid[1], mid[2],
        _camera_distance_text(camera_origin, object_center),
        fontsize=9, color="dimgray",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )
    scene_bounds = np.vstack([object_points, camera_origin[None, :]])
    _set_limits(ax1, scene_bounds, pad_ratio=0.08, min_pad=0.04)
    ax1.view_init(elev=18, azim=-70)
    ax1.set_title("Stage 1 — Patch selection in camera frame", fontsize=14)

    hm_vals = case["grasp_hm_values"]
    if hm_vals is not None and len(hm_vals) == len(grasp_points):
        ax2.scatter(
            grasp_points[:, 0], grasp_points[:, 1], grasp_points[:, 2],
            c=np.clip(hm_vals, 0.0, 1.0), cmap="turbo", s=9, alpha=0.55,
            vmin=0.0, vmax=1.0, depthshade=False,
        )
    else:
        ax2.scatter(
            grasp_points[:, 0], grasp_points[:, 1], grasp_points[:, 2],
            c="#7aa6d1", s=9, alpha=0.40, depthshade=False,
        )
    ax2.scatter(
        selected_patch_pts[:, 0], selected_patch_pts[:, 1], selected_patch_pts[:, 2],
        c="#ff8c00", s=18, alpha=0.95, depthshade=False,
    )
    ax2.scatter(seed_points[:, 0], seed_points[:, 1], seed_points[:, 2], c="black", s=18, alpha=0.85)
    centroid = chosen_sg.candidate.frame.centroid
    ax2.scatter(*centroid, c="black", s=80, zorder=20)
    ax2.text(
        centroid[0], centroid[1], centroid[2],
        f"  selected patch #{chosen_patch.patch_id}",
        fontsize=10, weight="bold", color="black",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )
    local_bounds = np.vstack([selected_patch_pts, seed_points, centroid[None, :]])
    _set_limits(ax2, local_bounds, pad_ratio=0.25, min_pad=0.015)
    ax2.view_init(elev=18, azim=-35)
    ax2.set_title("Selected local patch after FPS + heatmap sorting", fontsize=14)

    fig.suptitle(f"{case['prefix']} — grasp candidate generation", fontsize=18, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _export_stage2(case: dict, chosen_sg, out_path: Path) -> None:
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    for ax in (ax1, ax2):
        _style_3d(ax)

    object_points = case["object_points"]
    grasp_points = case["grasp_points"]
    camera_origin = case["camera_origin"]
    frame = chosen_sg.candidate.frame
    patch_pts = grasp_points[frame.patch.point_indices]
    contacts = [c.point for c in chosen_sg.candidate.contacts]
    contact_normals = [c.normal for c in chosen_sg.candidate.contacts]
    contact_arr = np.array(contacts, dtype=np.float64)
    triangle = np.vstack([contact_arr, contact_arr[0]])
    approach = _normalize(chosen_sg.candidate.approach_direction)

    for ax in (ax1, ax2):
        ax.scatter(
            object_points[:, 0], object_points[:, 1], object_points[:, 2],
            c="#c8c8c8", s=7, alpha=0.15, depthshade=False,
        )
        ax.scatter(
            patch_pts[:, 0], patch_pts[:, 1], patch_pts[:, 2],
            c="#ff9f0a", s=18, alpha=0.95, depthshade=False,
        )
        ax.plot(
            triangle[:, 0], triangle[:, 1], triangle[:, 2],
            linestyle="--", color="dimgray", linewidth=2.0, zorder=20,
        )
        _plot_contacts(ax, contacts, contact_normals, CONTACT_LABELS)

    centroid = frame.centroid
    t1 = _normalize(frame.tangent_1)
    t2 = _normalize(frame.tangent_2)
    n = _normalize(frame.normal)
    axis_len = 0.025
    ax1.scatter(*centroid, c="black", s=80, zorder=25)
    ax1.quiver(*centroid, *(t1 * axis_len), color="#ff3b30", linewidth=2.5, arrow_length_ratio=0.2)
    ax1.quiver(*centroid, *(t2 * axis_len), color="#34c759", linewidth=2.5, arrow_length_ratio=0.2)
    ax1.quiver(*centroid, *(n * axis_len), color="#0a84ff", linewidth=2.5, arrow_length_ratio=0.2)
    ax1.text(*(centroid + t1 * axis_len * 1.1), "t1", color="#ff3b30", fontsize=10, weight="bold")
    ax1.text(*(centroid + t2 * axis_len * 1.1), "t2", color="#34c759", fontsize=10, weight="bold")
    ax1.text(*(centroid + n * axis_len * 1.1), "n", color="#0a84ff", fontsize=10, weight="bold")
    grasp_center = contact_arr.mean(axis=0)
    app_start = grasp_center - approach * 0.03
    ax1.quiver(
        app_start[0], app_start[1], app_start[2],
        *(approach * 0.03),
        color="darkorange", linewidth=3.0, arrow_length_ratio=0.25,
    )
    ax1.text(
        app_start[0], app_start[1], app_start[2], "  approach",
        color="darkorange", fontsize=12, weight="bold",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )
    left_bounds = np.vstack([patch_pts, contact_arr, centroid[None, :]])
    _set_limits(ax1, left_bounds, pad_ratio=0.22, min_pad=0.02)
    _plot_camera_anchor(ax1, left_bounds)
    ax1.text2D(
        0.04, 0.04,
        _camera_distance_text(camera_origin, case["object_center"]),
        transform=ax1.transAxes,
        fontsize=10, color="dimgray",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )
    ax1.view_init(elev=18, azim=-60)
    ax1.set_title("Patch frame + approach direction", fontsize=14)

    zoom_bounds = np.vstack([patch_pts, contact_arr, centroid[None, :]])
    _set_limits(ax2, zoom_bounds, pad_ratio=0.18, min_pad=0.01)
    ax2.view_init(elev=18, azim=-35)
    ax2.set_title("Triangle grasp generated from 3 contacts", fontsize=14)

    fig.suptitle(f"{case['prefix']} — triangle grasp generation", fontsize=18, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _export_stage3(case: dict, chosen_sg, out_path: Path) -> None:
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    for ax in (ax1, ax2):
        _style_3d(ax)

    object_points = case["object_points"]
    camera_origin = case["camera_origin"]
    display_scored = case["scored"][:4]

    centers = np.array(
        [np.mean([ct.point for ct in sg.candidate.contacts], axis=0) for sg in display_scored],
        dtype=np.float64,
    )
    rank_colors = ["#ffd60a", "#30d158", "#64d2ff", "#bf5af2"]

    ax1.scatter(
        object_points[:, 0], object_points[:, 1], object_points[:, 2],
        c="#c8c8c8", s=7, alpha=0.16, depthshade=False,
    )
    for idx, (center, sg, color) in enumerate(zip(centers, display_scored, rank_colors), start=1):
        marker = "o" if sg.valid else "X"
        ax1.text(
            center[0], center[1], center[2],
            f"  rank {idx} | {sg.final_score:.3f}",
            fontsize=10, weight="bold", color="black",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        )
        ax1.scatter(
            center[0], center[1], center[2],
            c=color, s=180, marker=marker,
            edgecolors="black", linewidths=1.6, depthshade=False, zorder=20,
        )
    all_bounds = np.vstack([object_points, centers])
    _set_limits(ax1, all_bounds, pad_ratio=0.12, min_pad=0.02)
    _plot_camera_anchor(ax1, all_bounds)
    ax1.view_init(elev=18, azim=-50)
    ax1.set_title("Top candidates after scoring and constraints", fontsize=14)

    best_contacts = [ct.point for ct in chosen_sg.candidate.contacts]
    best_normals = [ct.normal for ct in chosen_sg.candidate.contacts]
    best_contact_arr = np.array(best_contacts, dtype=np.float64)
    tri = np.vstack([best_contact_arr, best_contact_arr[0]])
    ax2.scatter(
        object_points[:, 0], object_points[:, 1], object_points[:, 2],
        c="#d0d0d0", s=7, alpha=0.12, depthshade=False,
    )
    ax2.plot(tri[:, 0], tri[:, 1], tri[:, 2], linestyle="--", color="dimgray", linewidth=2.0)
    _plot_contacts(ax2, best_contacts, best_normals, CONTACT_LABELS)
    grasp_center = best_contact_arr.mean(axis=0)
    app = _normalize(chosen_sg.candidate.approach_direction)
    app_start = grasp_center - app * 0.025
    ax2.quiver(
        app_start[0], app_start[1], app_start[2],
        *(app * 0.025),
        color="darkorange", linewidth=3.0, arrow_length_ratio=0.25,
    )
    ax2.text(
        app_start[0], app_start[1], app_start[2], "  best grasp",
        color="darkorange", fontsize=12, weight="bold",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )
    zoom_bounds = np.vstack([best_contact_arr, object_points])
    _set_limits(ax2, zoom_bounds, pad_ratio=0.16, min_pad=0.01)
    ax2.view_init(elev=18, azim=-35)
    ax2.set_title("Best-ranked contact triangle", fontsize=14)

    fig.suptitle(f"{case['prefix']} — candidate ranking / optimization", fontsize=18, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 3 thesis-ready grasp stage figures.")
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=DEFAULT_CASE_DIR,
        help="Path to one grasp-ready case directory (default: blue_cup/primary_symmetric_3d).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where figures will be saved. Defaults to <case-dir>/thesis_stage_figures.",
    )
    args = parser.parse_args()

    case_dir = args.case_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else case_dir / "thesis_stage_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    case = _load_case(case_dir)
    chosen_sg = _match_best_scored(case)

    _export_stage1(case, chosen_sg, output_dir / "stage1_patch_selection.png")
    _export_stage2(case, chosen_sg, output_dir / "stage2_triangle_generation.png")
    _export_stage3(case, chosen_sg, output_dir / "stage3_candidate_ranking.png")

    print(f"Saved thesis figures to {output_dir}")


if __name__ == "__main__":
    main()
