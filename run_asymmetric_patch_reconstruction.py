from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.object_catalog import flexible_for_name, symmetry_for_name
from src.pipeline import PipelineConfig, SymmetryAwareSamPipeline
from src.reconstruction import CameraIntrinsics
from scipy.spatial import cKDTree

# ============================================================
# Basic helpers
# ============================================================

def safe_name(text: str) -> str:
    text = text.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-zA-Z0-9_\-.]+", "_", text)


def canonical_object_key(name: str) -> str:
    n = safe_name(name)

    aliases = {
        "black_computer_mouse": "computer_mouse",
        "white_computer_mouse": "computer_mouse",
        "mouse": "computer_mouse",
        "computer_mouse": "computer_mouse",

        "phillips_screwdriver": "screwdriver",
        "phillips_screw_driver": "screwdriver",
        "screw_driver": "screwdriver",
        "screwdriver": "screwdriver",
        "driver": "screwdriver",

        "toothpaste_box": "toothpaste_box",
        "soap_box": "soap_box",
        "juice_box": "juice_box",
        "cracker_box": "cracker_box",
        "cleansing_foam_tube": "cleansing_foam_tube",
        "shampoo_tube": "shampoo_bottle",
        "shampoo_bottle": "shampoo_bottle",
        "dove_bottle": "shampoo_bottle",
        "head_and_shoulders_bottle": "shampoo_bottle",
    }

    return aliases.get(n, n)


def is_cylindrical_fusion_object(object_key: str) -> bool:
    return any(
        token in object_key
        for token in ("cup", "bottle", "can", "tape", "roll", "tube")
    )


def load_depth_raw(path: Path, depth_unit: str) -> np.ndarray:
    suffix = path.suffix.lower()

    if suffix == ".npy":
        depth = np.load(path).astype(np.float32)
    elif suffix == ".npz":
        data = np.load(path)
        key = "depth" if "depth" in data else data.files[0]
        depth = data[key].astype(np.float32)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Could not read depth map: {path}")
        depth = depth.astype(np.float32)

    if depth.ndim == 3:
        depth = depth[..., 0]

    if depth_unit == "mm" or np.nanmax(depth) > 50.0:
        depth = depth / 1000.0

    return depth.astype(np.float32)


def prepare_depth_and_intrinsics(
    depth_path: Path,
    intrinsics_path: Path,
    target_size_hw: tuple[int, int],
    depth_unit: str,
) -> tuple[np.ndarray, CameraIntrinsics]:
    depth_raw = load_depth_raw(depth_path, depth_unit)
    intr_raw = CameraIntrinsics.from_json(intrinsics_path)

    raw_h, raw_w = depth_raw.shape[:2]
    target_h, target_w = target_size_hw

    if (raw_h, raw_w) != (target_h, target_w):
        depth = cv2.resize(depth_raw, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        depth = depth_raw

    if intr_raw.width is not None and intr_raw.height is not None:
        original_size = (int(intr_raw.height), int(intr_raw.width))
    else:
        original_size = (raw_h, raw_w)

    if original_size != (target_h, target_w):
        intr = intr_raw.scaled(original_size=original_size, target_size=target_size_hw)
    else:
        intr = intr_raw

    return depth.astype(np.float32), intr


def backproject(u: np.ndarray, v: np.ndarray, z: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    x = (u.astype(np.float32) - intr.cx) * z.astype(np.float32) / intr.fx
    y = (v.astype(np.float32) - intr.cy) * z.astype(np.float32) / intr.fy
    return np.stack([x, y, z.astype(np.float32)], axis=1).astype(np.float32)


def write_ply(path: Path, points_xyz: np.ndarray, colors_rgb: np.ndarray | None = None) -> None:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)

    if colors_rgb is None:
        colors_rgb = np.full((len(points_xyz), 3), 180, dtype=np.uint8)
    else:
        colors_rgb = np.asarray(colors_rgb, dtype=np.uint8)

    if len(points_xyz) != len(colors_rgb):
        raise ValueError(
            f"points_xyz and colors_rgb must have same length. "
            f"points={len(points_xyz)}, colors={len(colors_rgb)}, path={path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points_xyz, colors_rgb):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def heat_colors(values: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    base = np.asarray(color, dtype=np.float32)[None, :]

    if float(values.max() - values.min()) > 1e-6:
        alpha = ((values - values.min()) / (values.max() - values.min()))[:, None]
    else:
        alpha = np.ones((len(values), 1), dtype=np.float32)

    colors = 70.0 + alpha * (base - 70.0)
    return np.clip(colors, 0, 255).astype(np.uint8)


# ============================================================
# Pose helpers
# ============================================================

def euler_xyz_to_rotation_matrix(rx: float, ry: float, rz: float, degrees: bool = False) -> np.ndarray:
    if degrees:
        rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_mat = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry_mat = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz_mat = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)

    return (rz_mat @ ry_mat @ rx_mat).astype(np.float32)


def load_pose_matrix(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").strip()
    values = [float(x) for x in re.split(r"[\s,;]+", text) if x.strip()]

    if len(values) == 16:
        return np.asarray(values, dtype=np.float32).reshape(4, 4)

    if len(values) == 6:
        tx, ty, tz, rx, ry, rz = values

        # Important: pose rotations are radians.
        rot = euler_xyz_to_rotation_matrix(rx, ry, rz, degrees=False)

        t = np.eye(4, dtype=np.float32)
        t[:3, :3] = rot
        t[:3, 3] = np.asarray([tx, ty, tz], dtype=np.float32)
        return t

    raise RuntimeError(
        f"Pose file {path} must contain either 6 values or 16 values. Got {len(values)}."
    )


def transform_points(points_xyz: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)

    if len(points_xyz) == 0:
        return points_xyz

    ones = np.ones((len(points_xyz), 1), dtype=np.float32)
    homo = np.hstack([points_xyz, ones])
    transformed = (transform_4x4 @ homo.T).T[:, :3]

    return transformed.astype(np.float32)


# ============================================================
# Pointcloud filtering + alignment
# ============================================================

def filter_pointcloud_outliers(
    points: np.ndarray,
    heat_values: np.ndarray | None = None,
    keep_percentile: float = 95.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    points = np.asarray(points, dtype=np.float32)

    if len(points) < 20 or keep_percentile >= 100.0:
        return points, heat_values

    center = np.median(points, axis=0)
    dist = np.linalg.norm(points - center[None, :], axis=1)

    threshold = np.percentile(dist, keep_percentile)
    keep = dist <= threshold

    points_f = points[keep].astype(np.float32)

    if heat_values is None:
        return points_f, None

    return points_f, heat_values[keep].astype(np.float32)

def radius_outlier_filter(
    points: np.ndarray,
    heat_values: np.ndarray | None = None,
    radius: float = 0.008,
    min_neighbors: int = 10,
) -> tuple[np.ndarray, np.ndarray | None]:
    points = np.asarray(points, dtype=np.float32)

    if len(points) < min_neighbors:
        return points, heat_values

    tree = cKDTree(points)

    counts = np.asarray(
        [len(tree.query_ball_point(p, radius)) for p in points],
        dtype=np.int32,
    )

    keep = counts >= min_neighbors
    points_f = points[keep].astype(np.float32)

    if heat_values is None:
        return points_f, None

    return points_f, heat_values[keep].astype(np.float32)

def align_elongated_patch_180_y(patch1: np.ndarray, patch2: np.ndarray) -> np.ndarray:
    """
    Pentru screwdriver / obiecte alungite.
    Rotește patch2 cu 180° pe Y și îl pune peste centroidul patch1.
    """
    patch1 = np.asarray(patch1, dtype=np.float32)
    patch2 = np.asarray(patch2, dtype=np.float32)

    if len(patch1) == 0 or len(patch2) == 0:
        return patch2

    c1 = np.median(patch1, axis=0)
    c2 = np.median(patch2, axis=0)

    p2 = patch2 - c2[None, :]

    # 180 degrees around Y axis: X/Z flip, Y stays.
    p2[:, 0] *= -1.0
    p2[:, 2] *= -1.0

    return (p2 + c1[None, :]).astype(np.float32)


def align_compact_centroid_only(patch1: np.ndarray, patch2: np.ndarray) -> np.ndarray:
    """
    Pentru mouse / obiecte compacte.
    Nu forțăm rotație agresivă; doar mutăm centroidul view2 peste view1.
    """
    patch1 = np.asarray(patch1, dtype=np.float32)
    patch2 = np.asarray(patch2, dtype=np.float32)

    if len(patch1) == 0 or len(patch2) == 0:
        return patch2

    c1 = np.median(patch1, axis=0)
    c2 = np.median(patch2, axis=0)

    return (patch2 - c2[None, :] + c1[None, :]).astype(np.float32)


def best_fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)

    src_centered = source - src_centroid[None, :]
    tgt_centered = target - tgt_centroid[None, :]

    h = src_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T

    t = tgt_centroid - (r @ src_centroid)
    return r.astype(np.float32), t.astype(np.float32)


def apply_rigid_transform(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return points
    return (points @ rotation.T + translation[None, :]).astype(np.float32)


def align_patch_icp_world(
    patch1: np.ndarray,
    patch2: np.ndarray,
    *,
    max_iterations: int = 25,
    trim_percentile: float = 85.0,
    max_pairs: int = 3000,
) -> np.ndarray:
    """Align view2 to view1 with a small rigid ICP refinement in world coordinates."""
    patch1 = np.asarray(patch1, dtype=np.float32)
    patch2 = np.asarray(patch2, dtype=np.float32)

    if len(patch1) < 6 or len(patch2) < 6:
        return align_compact_centroid_only(patch1, patch2)

    c1 = np.median(patch1, axis=0)
    c2 = np.median(patch2, axis=0)
    aligned = (patch2 - c2[None, :] + c1[None, :]).astype(np.float32)

    ref = patch1
    if len(ref) > max_pairs:
        ref_idx = np.linspace(0, len(ref) - 1, max_pairs).astype(np.int32)
        ref = ref[ref_idx]

    tree = cKDTree(ref)

    for _ in range(max_iterations):
        moving = aligned
        if len(moving) > max_pairs:
            mov_idx = np.linspace(0, len(moving) - 1, max_pairs).astype(np.int32)
            moving_sample = moving[mov_idx]
        else:
            mov_idx = None
            moving_sample = moving

        distances, indices = tree.query(moving_sample, k=1)
        if len(distances) < 6:
            break

        cutoff = np.percentile(distances, trim_percentile)
        keep = distances <= cutoff
        if int(keep.sum()) < 6:
            break

        src = moving_sample[keep]
        dst = ref[indices[keep]]
        rotation, translation = best_fit_rigid_transform(src, dst)
        aligned_next = apply_rigid_transform(aligned, rotation, translation)

        mean_step = float(np.linalg.norm(aligned_next - aligned, axis=1).mean())
        aligned = aligned_next
        if mean_step < 1e-5:
            break

    return aligned.astype(np.float32)


def align_patch_by_object_type(
    object_key: str,
    patch1: np.ndarray,
    patch2_raw: np.ndarray,
    heat1: np.ndarray,
    heat2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Returnează:
      patch1_filtered, patch2_aligned_filtered, heat1_filtered, heat2_filtered, method
    """

    if object_key in {"blue_cup", "roll_of_tape"}:
        patch1, heat1 = filter_pointcloud_outliers(patch1, heat1, keep_percentile=95.0)
        patch2_raw, heat2 = filter_pointcloud_outliers(patch2_raw, heat2, keep_percentile=95.0)
        patch2_aligned = align_elongated_patch_180_y(patch1, patch2_raw)
        method = "legacy_180_y_centroid_outlier_filtered"

    elif is_cylindrical_fusion_object(object_key):
        patch1, heat1 = filter_pointcloud_outliers(patch1, heat1, keep_percentile=97.0)
        patch2_raw, heat2 = filter_pointcloud_outliers(patch2_raw, heat2, keep_percentile=97.0)
        patch1, heat1 = radius_outlier_filter(
            patch1,
            heat1,
            radius=0.008,
            min_neighbors=8,
        )
        patch2_raw, heat2 = radius_outlier_filter(
            patch2_raw,
            heat2,
            radius=0.008,
            min_neighbors=8,
        )
        patch2_aligned = align_compact_centroid_only(patch1, patch2_raw)
        method = "cylindrical_centroid_only_outlier_filtered"

    elif object_key in {"computer_mouse", "mouse"}:
        patch1, heat1 = filter_pointcloud_outliers(patch1, heat1, keep_percentile=85.0)
        patch2_raw, heat2 = filter_pointcloud_outliers(patch2_raw, heat2, keep_percentile=85.0)
        patch1, heat1 = radius_outlier_filter(
            patch1,
            heat1,
            radius=0.006,
            min_neighbors=12,
        )

        patch2_raw, heat2 = radius_outlier_filter(
            patch2_raw,
            heat2,
            radius=0.006,
            min_neighbors=12,
        )
        patch2_aligned = align_patch_icp_world(patch1, patch2_raw, trim_percentile=80.0)
        method = "compact_world_icp_outlier_filtered"

    elif object_key in {"screwdriver", "marker", "pen", "toothbrush"}:
        patch1, heat1 = filter_pointcloud_outliers(patch1, heat1, keep_percentile=97.0)
        patch2_raw, heat2 = filter_pointcloud_outliers(patch2_raw, heat2, keep_percentile=97.0)
        patch1, heat1 = radius_outlier_filter(
            patch1,
            heat1,
            radius=0.018,
            min_neighbors=4,
        )

        patch2_raw, heat2 = radius_outlier_filter(
            patch2_raw,
            heat2,
            radius=0.018,
            min_neighbors=4,
        )
        patch2_aligned = align_patch_icp_world(patch1, patch2_raw, trim_percentile=90.0)
        method = "elongated_world_icp_outlier_filtered"

    else:
        patch1, heat1 = filter_pointcloud_outliers(patch1, heat1, keep_percentile=95.0)
        patch2_raw, heat2 = filter_pointcloud_outliers(patch2_raw, heat2, keep_percentile=95.0)

        patch2_aligned = align_patch_icp_world(patch1, patch2_raw, trim_percentile=85.0)
        method = "default_world_icp_outlier_filtered"

    return (
        patch1.astype(np.float32),
        patch2_aligned.astype(np.float32),
        heat1.astype(np.float32),
        heat2.astype(np.float32),
        method,
    )


# ============================================================
# Mask / patch reconstruction
# ============================================================

def object_depth_filter(
    mask: np.ndarray,
    depth_m: np.ndarray,
    max_depth_m: float,
    depth_window_m: float,
) -> np.ndarray:
    mask = mask.astype(bool)

    valid_global = (
        np.isfinite(depth_m)
        & (depth_m > 0.0)
        & (depth_m <= max_depth_m)
    )

    # Pentru asimetrice păstrăm toate punctele valide din mască.
    return mask & valid_global


def build_patch_binary(
    *,
    heatmap: np.ndarray,
    mask: np.ndarray,
    obj_valid: np.ndarray,
    heatmap_threshold: float,
    patch_source: str,
    patch_dilate_kernel: int,
    patch_dilate_iterations: int,
) -> np.ndarray:
    if patch_source == "mask":
        patch_binary = mask & obj_valid
    elif patch_source == "heatmap":
        patch_binary = (heatmap >= heatmap_threshold) & mask & obj_valid
    else:
        raise ValueError(f"Unknown patch_source: {patch_source}")

    if patch_dilate_kernel > 1 and patch_dilate_iterations > 0:
        k = int(patch_dilate_kernel)
        if k % 2 == 0:
            k += 1

        kernel = np.ones((k, k), dtype=np.uint8)

        patch_binary = cv2.dilate(
            patch_binary.astype(np.uint8),
            kernel,
            iterations=int(patch_dilate_iterations),
        ).astype(bool)

        patch_binary = patch_binary & mask & obj_valid

    return patch_binary.astype(bool)


def connected_components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    comps: list[np.ndarray] = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            comps.append(labels == label)

    comps.sort(key=lambda c: int(c.sum()), reverse=True)
    return comps


def mask_bbox_aspect(mask: np.ndarray) -> float:
    ys, xs = np.where(mask.astype(bool))

    if len(xs) == 0:
        return 0.0

    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)

    if height <= 0.0:
        return 0.0

    return width / height


def reconstruct_view_object(
    *,
    object_name: str,
    rgb: np.ndarray,
    mask: np.ndarray,
    heatmap: np.ndarray,
    depth_m: np.ndarray,
    intr: CameraIntrinsics,
    pose_cam_to_world: np.ndarray,
    heatmap_threshold: float,
    min_component_area: int,
    max_depth_m: float,
    depth_window_m: float,
    patch_source: str,
    patch_dilate_kernel: int,
    patch_dilate_iterations: int,
) -> dict[str, Any]:
    mask = mask.astype(bool)
    heatmap = np.asarray(heatmap, dtype=np.float32)

    obj_valid = object_depth_filter(
        mask=mask,
        depth_m=depth_m,
        max_depth_m=max_depth_m,
        depth_window_m=depth_window_m,
    )

    ov, ou = np.where(obj_valid)

    if len(ou) == 0:
        raise RuntimeError(f"No valid object depth for {object_name}")

    object_camera_xyz = backproject(ou, ov, depth_m[ov, ou], intr)
    object_world_xyz = transform_points(object_camera_xyz, pose_cam_to_world)
    object_rgb = rgb[ov, ou].astype(np.uint8)

    patch_binary = build_patch_binary(
        heatmap=heatmap,
        mask=mask,
        obj_valid=obj_valid,
        heatmap_threshold=heatmap_threshold,
        patch_source=patch_source,
        patch_dilate_kernel=patch_dilate_kernel,
        patch_dilate_iterations=patch_dilate_iterations,
    )

    components = connected_components(patch_binary, min_component_area)

    patch_world_list = []
    heat_values_list = []
    patch_pixels_list = []

    for comp in components:
        pv, pu = np.where(comp)

        if len(pu) == 0:
            continue

        patch_camera_xyz = backproject(pu, pv, depth_m[pv, pu], intr)
        patch_world_xyz = transform_points(patch_camera_xyz, pose_cam_to_world)
        heat_values = heatmap[pv, pu].astype(np.float32)

        patch_world_list.append(patch_world_xyz)
        heat_values_list.append(heat_values)
        patch_pixels_list.append(np.stack([pu, pv], axis=1).astype(np.int32))

    if patch_world_list:
        patch_world_xyz = np.vstack(patch_world_list).astype(np.float32)
        patch_heat_values = np.concatenate(heat_values_list).astype(np.float32)
        patch_pixels_uv = np.vstack(patch_pixels_list).astype(np.int32)
    else:
        patch_world_xyz = np.zeros((0, 3), dtype=np.float32)
        patch_heat_values = np.zeros((0,), dtype=np.float32)
        patch_pixels_uv = np.zeros((0, 2), dtype=np.int32)

    overlay = rgb.copy()

    overlay[mask] = (
        0.5 * overlay[mask].astype(np.float32)
        + 0.5 * np.array([80, 80, 255], dtype=np.float32)
    ).astype(np.uint8)

    overlay[patch_binary] = (
        0.5 * overlay[patch_binary].astype(np.float32)
        + 0.5 * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)

    return {
        "object_name": object_name,
        "object_key": canonical_object_key(object_name),
        "object_flexible": bool(flexible_for_name(object_name)),
        "object_world_xyz": object_world_xyz,
        "object_rgb": object_rgb,
        "patch_world_xyz": patch_world_xyz,
        "patch_heat_values": patch_heat_values,
        "patch_pixels_uv": patch_pixels_uv,
        "overlay_rgb": overlay,
        "num_components": len(components),
    }


def collect_asymmetric_view_results(
    *,
    pipeline_results: list[Any],
    depth_m: np.ndarray,
    intr: CameraIntrinsics,
    pose_cam_to_world: np.ndarray,
    heatmap_threshold: float,
    min_component_area: int,
    max_depth_m: float,
    depth_window_m: float,
    patch_source: str,
    patch_dilate_kernel: int,
    patch_dilate_iterations: int,
) -> dict[str, list[dict[str, Any]]]:
    collected: dict[str, list[dict[str, Any]]] = {}

    for r in pipeline_results:
        #if symmetry_for_name(r.object_name) != "asymmetric":
         #   continue
        object_key = canonical_object_key(r.object_name)

        try:
            item = reconstruct_view_object(
                object_name=r.object_name,
                rgb=r.rgb_resized,
                mask=r.sam_mask,
                heatmap=r.final_heatmap,
                depth_m=depth_m,
                intr=intr,
                pose_cam_to_world=pose_cam_to_world,
                heatmap_threshold=heatmap_threshold,
                min_component_area=min_component_area,
                max_depth_m=max_depth_m,
                depth_window_m=depth_window_m,
                patch_source=patch_source,
                patch_dilate_kernel=patch_dilate_kernel,
                patch_dilate_iterations=patch_dilate_iterations,
            )

            key = item["object_key"]
            collected.setdefault(key, []).append(item)

            print(
                f"[VIEW OBJECT] {r.object_name}: "
                f"object_points={len(item['object_world_xyz'])}, "
                f"patch_points={len(item['patch_world_xyz'])}"
            )

        except Exception as exc:
            print(f"[WARNING] Failed view reconstruction for {r.object_name!r}: {exc}")

    return collected


# ============================================================
# Save / visualize
# ============================================================

def save_fused_object(
    *,
    output_dir: Path,
    object_key: str,
    view1_item: dict[str, Any],
    view2_item: dict[str, Any],
    image1_name: str,
    image2_name: str,
    pose1: np.ndarray,
    pose2: np.ndarray,
    intr: CameraIntrinsics,
) -> Path:
    obj_dir = output_dir / object_key
    obj_dir.mkdir(parents=True, exist_ok=True)
    object_flexible = bool(
        view1_item.get("object_flexible", False)
        or view2_item.get("object_flexible", False)
    )

    obj1 = view1_item["object_world_xyz"]
    obj2 = view2_item["object_world_xyz"]
    rgb1 = view1_item["object_rgb"]
    rgb2 = view2_item["object_rgb"]

    patch1_raw = view1_item["patch_world_xyz"]
    patch2_raw = view2_item["patch_world_xyz"]
    heat1_raw = view1_item["patch_heat_values"]
    heat2_raw = view2_item["patch_heat_values"]

    patch1, patch2_aligned, heat1, heat2, alignment_method = align_patch_by_object_type(
        object_key=object_key,
        patch1=patch1_raw,
        patch2_raw=patch2_raw,
        heat1=heat1_raw,
        heat2=heat2_raw,
    )

    if len(patch1) + len(patch2_aligned) > 0:
        fused_patch = np.vstack([patch1, patch2_aligned]).astype(np.float32)
        fused_heat = np.concatenate([heat1, heat2]).astype(np.float32)
    else:
        fused_patch = np.zeros((0, 3), dtype=np.float32)
        fused_heat = np.zeros((0,), dtype=np.float32)

    # Save raw object clouds.
    write_ply(obj_dir / "view1_object_world.ply", obj1, rgb1)
    write_ply(obj_dir / "view2_object_world_raw.ply", obj2, rgb2)

    # Save patches.
    write_ply(obj_dir / "view1_patch_world_raw.ply", patch1_raw, heat_colors(heat1_raw, (0, 220, 0)))
    write_ply(obj_dir / "view2_patch_world_raw.ply", patch2_raw, heat_colors(heat2_raw, (0, 220, 255)))

    write_ply(obj_dir / "view1_patch_world_filtered.ply", patch1, heat_colors(heat1, (0, 220, 0)))
    write_ply(obj_dir / "view2_patch_world_aligned.ply", patch2_aligned, heat_colors(heat2, (0, 220, 255)))
    write_ply(obj_dir / "fused_patch_world_aligned.ply", fused_patch, heat_colors(fused_heat, (255, 165, 0)))

    # Combined clean visualization cloud.
    combined_points = []
    combined_colors = []

    if len(patch1) > 0:
        combined_points.append(patch1)
        combined_colors.append(np.full((len(patch1), 3), [0, 220, 0], dtype=np.uint8))

    if len(patch2_aligned) > 0:
        combined_points.append(patch2_aligned)
        combined_colors.append(np.full((len(patch2_aligned), 3), [0, 220, 255], dtype=np.uint8))

    if combined_points:
        write_ply(
            obj_dir / "combined_aligned_patches.ply",
            np.vstack(combined_points),
            np.vstack(combined_colors),
        )

    cv2.imwrite(
        str(obj_dir / "view1_mask_patch_overlay.png"),
        cv2.cvtColor(view1_item["overlay_rgb"], cv2.COLOR_RGB2BGR),
    )

    cv2.imwrite(
        str(obj_dir / "view2_mask_patch_overlay.png"),
        cv2.cvtColor(view2_item["overlay_rgb"], cv2.COLOR_RGB2BGR),
    )

    np.savez_compressed(
        obj_dir / "fused_data.npz",
        view1_object_world_xyz=obj1.astype(np.float32),
        view2_object_world_xyz=obj2.astype(np.float32),
        view1_patch_world_raw_xyz=patch1_raw.astype(np.float32),
        view2_patch_world_raw_xyz=patch2_raw.astype(np.float32),
        view1_patch_world_xyz=patch1.astype(np.float32),
        view2_patch_world_xyz=patch2_aligned.astype(np.float32),
        fused_patch_world_xyz=fused_patch.astype(np.float32),
        view1_heat_values=heat1.astype(np.float32),
        view2_heat_values=heat2.astype(np.float32),
        fused_heat_values=fused_heat.astype(np.float32),
        object_flexible=np.asarray(object_flexible, dtype=np.bool_),
    )

    metadata = {
        "object_key": object_key,
        "view1_object_name": view1_item["object_name"],
        "view2_object_name": view2_item["object_name"],
        "flexible": object_flexible,
        "object_flexible": object_flexible,
        "view1_image": image1_name,
        "view2_image": image2_name,
        "alignment_method": alignment_method,
        "num_view1_object_points": int(len(obj1)),
        "num_view2_object_points": int(len(obj2)),
        "num_view1_patch_points_raw": int(len(patch1_raw)),
        "num_view2_patch_points_raw": int(len(patch2_raw)),
        "num_view1_patch_points_filtered": int(len(patch1)),
        "num_view2_patch_points_aligned": int(len(patch2_aligned)),
        "num_fused_patch_points": int(len(fused_patch)),
        "pose1_cam_to_world": pose1.astype(float).tolist(),
        "pose2_cam_to_world": pose2.astype(float).tolist(),
        "intrinsics": asdict(intr),
    }

    (obj_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return obj_dir


def visualize_fused_object(object_dir: Path, show: bool = False) -> None:
    data_path = object_dir / "fused_data.npz"

    if not data_path.exists():
        print(f"[WARNING] Missing fused data: {data_path}")
        return

    data = np.load(data_path)

    p1 = data["view1_patch_world_xyz"].astype(np.float32)
    p2 = data["view2_patch_world_xyz"].astype(np.float32)

    p1 = p1[np.isfinite(p1).all(axis=1)]
    p2 = p2[np.isfinite(p2).all(axis=1)]

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection="3d")

    if len(p1) > 0:
        ax.scatter(p1[:, 0], p1[:, 1], p1[:, 2], s=12, c="limegreen", alpha=1.0, label="view1 patch")

    if len(p2) > 0:
        ax.scatter(p2[:, 0], p2[:, 1], p2[:, 2], s=8, c="cyan", alpha=0.75, label="view2 patch aligned")

    all_parts = [x for x in [p1, p2] if len(x) > 0]

    if not all_parts:
        plt.close(fig)
        return

    all_pts = np.vstack(all_parts)
    center = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2

    if max_range < 1e-4:
        max_range = 0.01

    max_range *= 1.15

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.set_xlabel("World X")
    ax.set_ylabel("World Y")
    ax.set_zlabel("World Z")

    ax.set_title(f"{object_dir.name}: asymmetric patch fusion")
    ax.legend(fontsize=10)

    ax.view_init(elev=-90, azim=-90)

    plt.tight_layout()

    out_path = object_dir / "aligned_patches_visualization_3d.png"
    plt.savefig(out_path, dpi=180)

    if show:
        plt.show()

    plt.close(fig)

    print(f"Saved visualization to: {out_path}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-view asymmetric patch reconstruction.")

    parser.add_argument("--heatmap-model", required=True, type=Path)

    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--second-input", required=True, type=Path)

    parser.add_argument("--depth", required=True, type=Path)
    parser.add_argument("--second-depth", required=True, type=Path)

    parser.add_argument("--intrinsics", required=True, type=Path)

    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--second-pose", required=True, type=Path)

    parser.add_argument("--pose-convention", choices=["camera_to_world", "world_to_camera"], default="world_to_camera")

    parser.add_argument("--output-dir", type=Path, default=Path("outputs_asymmetric_patch_reconstruction"))

    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--depth-unit", choices=["m", "mm"], default="m")
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--depth-window", type=float, default=0.12)

    parser.add_argument("--patch-source", choices=["heatmap", "mask"], default="mask")
    parser.add_argument("--heatmap-threshold", type=float, default=0.35)
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--patch-dilate-kernel", type=int, default=0)
    parser.add_argument("--patch-dilate-iterations", type=int, default=0)

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-pipeline-images", action="store_true")

    parser.add_argument("--detector-backend", type=str, default="grounding-dino", choices=["grounding-dino", "owlvit"])
    parser.add_argument("--detector-model-id", type=str, default=None)
    parser.add_argument("--detector-threshold", type=float, default=0.10)
    parser.add_argument("--detector-text-threshold", type=float, default=0.20)
    parser.add_argument("--detector-label-batch-size", type=int, default=20)
    parser.add_argument("--detector-top-k", type=int, default=None)

    parser.add_argument("--sam-model-id", type=str, default="facebook/sam3")
    parser.add_argument("--sam-score-threshold", type=float, default=0.5)
    parser.add_argument("--sam-mask-threshold", type=float, default=0.5)
    parser.add_argument("--mask-background", type=str, default="black", choices=["black", "white"])
    parser.add_argument("--overlay-alpha", type=float, default=0.45)

    parser.add_argument("--postprocess-edges", action="store_true")
    parser.add_argument("--edge-shift-strength", type=float, default=0.65)
    parser.add_argument("--edge-max-distance", type=float, default=50.0)
    parser.add_argument("--heat-threshold", type=float, default=0.1)
    parser.add_argument("--post-smooth-sigma", type=float, default=1.6)
    parser.add_argument("--canny-low", type=int, default=50)
    parser.add_argument("--canny-high", type=int, default=150)
    parser.add_argument("--preserve-original", type=float, default=0.25)

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    paths = [
        args.heatmap_model,
        args.input,
        args.second_input,
        args.depth,
        args.second_depth,
        args.intrinsics,
        args.pose,
        args.second_pose,
    ]

    for p in paths:
        if not p.exists():
            raise RuntimeError(f"Path not found: {p}")


def build_pipeline(args: argparse.Namespace) -> SymmetryAwareSamPipeline:
    config = PipelineConfig(
        image_size=(args.height, args.width),
        device=args.device,
        detector_model_id=args.detector_model_id or (
            "IDEA-Research/grounding-dino-base"
            if args.detector_backend == "grounding-dino"
            else "google/owlvit-base-patch32"
        ),
        detector_backend=args.detector_backend,
        detector_threshold=args.detector_threshold,
        detector_text_threshold=args.detector_text_threshold,
        detector_label_batch_size=args.detector_label_batch_size,
        detector_top_k=args.detector_top_k,
        sam_model_id=args.sam_model_id,
        sam_score_threshold=args.sam_score_threshold,
        sam_mask_threshold=args.sam_mask_threshold,
        heatmap_threshold=args.heatmap_threshold,
        overlay_alpha=args.overlay_alpha,
        mask_background=args.mask_background,
        postprocess_edges=args.postprocess_edges,
        edge_shift_strength=args.edge_shift_strength,
        edge_max_distance=args.edge_max_distance,
        heat_threshold=args.heat_threshold,
        post_smooth_sigma=args.post_smooth_sigma,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        preserve_original=args.preserve_original,
        depth_path=None,
        intrinsics_path=None,
    )

    return SymmetryAwareSamPipeline(args.heatmap_model, config)


def main() -> None:
    args = parse_args()
    validate_args(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_size = (args.height, args.width)

    depth1_m, intr = prepare_depth_and_intrinsics(
        args.depth,
        args.intrinsics,
        image_size,
        args.depth_unit,
    )

    depth2_m, _ = prepare_depth_and_intrinsics(
        args.second_depth,
        args.intrinsics,
        image_size,
        args.depth_unit,
    )

    pose1_raw = load_pose_matrix(args.pose)
    pose2_raw = load_pose_matrix(args.second_pose)

    if args.pose_convention == "world_to_camera":
        pose1 = np.linalg.inv(pose1_raw).astype(np.float32)
        pose2 = np.linalg.inv(pose2_raw).astype(np.float32)
    else:
        pose1 = pose1_raw.astype(np.float32)
        pose2 = pose2_raw.astype(np.float32)

    print("Pose convention:", args.pose_convention)
    print("Pose 1 camera center:", pose1[:3, 3])
    print("Pose 2 camera center:", pose2[:3, 3])

    pipeline = build_pipeline(args)

    print("[RUN] view 1")
    predictions1, results1 = pipeline.run(args.input, second_input_image=None)
    print("Predictions view 1:")
    print(predictions1)

    print("[RUN] view 2")
    predictions2, results2 = pipeline.run(args.second_input, second_input_image=None)
    print("Predictions view 2:")
    print(predictions2)

    if args.save_pipeline_images:
        view1_dir = args.output_dir / "pipeline_view1"
        view2_dir = args.output_dir / "pipeline_view2"

        for r in results1:
            r.save(view1_dir)

        for r in results2:
            r.save(view2_dir)

        print(f"Saved pipeline images to: {view1_dir}")
        print(f"Saved pipeline images to: {view2_dir}")

    print("[RECONSTRUCT] view 1 asymmetric objects")
    view1_objects = collect_asymmetric_view_results(
        pipeline_results=results1,
        depth_m=depth1_m,
        intr=intr,
        pose_cam_to_world=pose1,
        heatmap_threshold=args.heatmap_threshold,
        min_component_area=args.min_component_area,
        max_depth_m=args.max_depth,
        depth_window_m=args.depth_window,
        patch_source=args.patch_source,
        patch_dilate_kernel=args.patch_dilate_kernel,
        patch_dilate_iterations=args.patch_dilate_iterations,
    )

    print("[RECONSTRUCT] view 2 asymmetric objects")
    view2_objects = collect_asymmetric_view_results(
        pipeline_results=results2,
        depth_m=depth2_m,
        intr=intr,
        pose_cam_to_world=pose2,
        heatmap_threshold=args.heatmap_threshold,
        min_component_area=args.min_component_area,
        max_depth_m=args.max_depth,
        depth_window_m=args.depth_window,
        patch_source=args.patch_source,
        patch_dilate_kernel=args.patch_dilate_kernel,
        patch_dilate_iterations=args.patch_dilate_iterations,
    )

    common_keys = sorted(set(view1_objects.keys()) & set(view2_objects.keys()))
    print(f"Common asymmetric object keys: {common_keys}")

    fused_count = 0

    fused_root = args.output_dir / "fused_asymmetric_objects"
    fused_root.mkdir(parents=True, exist_ok=True)

    for key in common_keys:
        item1 = max(view1_objects[key], key=lambda x: len(x["patch_world_xyz"]))
        item2 = max(view2_objects[key], key=lambda x: len(x["patch_world_xyz"]))

        try:
            print(f"[FUSE] {key}")

            obj_dir = save_fused_object(
                output_dir=fused_root,
                object_key=key,
                view1_item=item1,
                view2_item=item2,
                image1_name=str(args.input),
                image2_name=str(args.second_input),
                pose1=pose1,
                pose2=pose2,
                intr=intr,
            )

            if args.visualize:
                visualize_fused_object(obj_dir, show=args.show)

            fused_count += 1

        except Exception as exc:
            print(f"[WARNING] Failed fusion for {key!r}: {exc}")

    print(f"Done. Fused {fused_count}/{len(common_keys)} asymmetric object(s).")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
