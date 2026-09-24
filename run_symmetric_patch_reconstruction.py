from __future__ import annotations

"""
run_symmetric_grasp_reconstruction_semantic_cylinder.py

Variantă finală pentru obiecte simetrice.

Ideea:
  - pentru cup/mug/can/bottle/glass/jar: obiectul este aproximat semantic ca cilindru;
  - patch-ul original 3D este rotit 180 grade în jurul axei verticale estimate a cilindrului;
  - pentru roll_of_tape/tape: patch-ul este rotit 180 grade în jurul centrului estimat al rolei;
  - pentru alte obiecte simetrice: fallback 2D mirror + backprojection.

Important:
  - rețeaua produce heatmap, nu patch explicit;
  - patch-ul este construit din heatmap + mask + depth valid;
  - poți alege dacă patch-ul vine din heatmap sau din toată masca.
"""

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.object_catalog import flexible_for_name, symmetry_for_name
from src.pipeline import PipelineConfig, SymmetryAwareSamPipeline
from src.reconstruction import CameraIntrinsics


CYLINDER_OBJECT_KEYS = [
    "cup", "mug", "can", "bottle", "glass", "jar", "vase", "bowl", "tube"
]

ROLL_OBJECT_KEYS = [
    "roll_of_tape",
    "tape",
]


def safe_name(text: str) -> str:
    text = text.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-zA-Z0-9_\-.]+", "_", text)


def is_cylinder_like(name: str) -> bool:
    n = name.lower().replace(" ", "_")
    return any(k in n for k in CYLINDER_OBJECT_KEYS)


def is_roll_like(name: str) -> bool:
    n = name.lower().replace(" ", "_")
    return any(k in n for k in ROLL_OBJECT_KEYS)


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

    finite_max = float(np.nanmax(depth[np.isfinite(depth)])) if np.any(np.isfinite(depth)) else 0.0
    if depth_unit == "mm" or finite_max > 50.0:
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
        raise ValueError("points_xyz and colors_rgb must have the same length")

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


def filter_patch_3d_by_centroid(
    points_xyz: np.ndarray,
    heat_values: np.ndarray,
    keep_percentile: float = 85.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Elimină outlierii 3D din patch.
    Dacă keep_percentile >= 100, nu filtrează nimic.
    Returnează points, heat_values, keep_mask.
    """

    if len(points_xyz) < 20 or keep_percentile >= 100.0:
        keep = np.ones(len(points_xyz), dtype=bool)
        return points_xyz, heat_values, keep

    centroid = np.median(points_xyz, axis=0)
    distances = np.linalg.norm(points_xyz - centroid[None, :], axis=1)

    threshold = np.percentile(distances, keep_percentile)
    keep = distances <= threshold

    return points_xyz[keep], heat_values[keep], keep


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

    rough = mask & valid_global
    z_values = depth_m[rough]

    if len(z_values) == 0:
        return rough

    z_med = float(np.median(z_values))

    filtered = (
        valid_global
        & (depth_m >= z_med - depth_window_m)
        & (depth_m <= z_med + depth_window_m)
    )

    return mask & filtered


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
    """
    Construiește regiunea 2D care va deveni pointcloud.

    patch_source:
      - heatmap: folosește heatmap >= threshold
      - mask: folosește toată masca validă în depth

    Dilatarea extinde patch-ul către marginile obiectului,
    dar îl limitează mereu la mask & obj_valid.
    """

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

        kernel = np.ones((k, k), np.uint8)

        patch_binary = cv2.dilate(
            patch_binary.astype(np.uint8),
            kernel,
            iterations=int(patch_dilate_iterations),
        ).astype(bool)

        patch_binary = patch_binary & mask & obj_valid

    return patch_binary.astype(bool)


# ---------------------------------------------------------------------
# Semantic cylinder model
# ---------------------------------------------------------------------

def estimate_semantic_cylinder(
    mask: np.ndarray,
    depth_m: np.ndarray,
    obj_valid: np.ndarray,
    intr: CameraIntrinsics,
    center_depth_offset_ratio: float,
) -> dict[str, Any]:
    ys, xs = np.where(obj_valid)

    if len(xs) == 0:
        raise RuntimeError("No valid object depth for cylinder estimation")

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    u_center = 0.5 * (x1 + x2)
    v_center = 0.5 * (y1 + y2)

    z_visible = float(np.median(depth_m[ys, xs]))

    pixel_radius = 0.5 * float(x2 - x1)
    radius_m = pixel_radius * z_visible / float(intr.fx)

    z_center = z_visible + center_depth_offset_ratio * radius_m

    center_xyz = backproject(
        np.asarray([u_center], dtype=np.float32),
        np.asarray([v_center], dtype=np.float32),
        np.asarray([z_center], dtype=np.float32),
        intr,
    )[0].astype(np.float32)

    y_minmax_pts = backproject(
        np.asarray([u_center, u_center], dtype=np.float32),
        np.asarray([y1, y2], dtype=np.float32),
        np.asarray([z_center, z_center], dtype=np.float32),
        intr,
    )

    axis_direction_xyz = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "center_uv": [float(u_center), float(v_center)],
        "z_visible_m": float(z_visible),
        "radius_m": float(radius_m),
        "z_center_m": float(z_center),
        "center_xyz": center_xyz.astype(float).tolist(),
        "axis_direction_xyz": axis_direction_xyz.astype(float).tolist(),
        "axis_segment_xyz": y_minmax_pts.astype(float).tolist(),
        "center_depth_offset_ratio": float(center_depth_offset_ratio),
    }


def semantic_cylinder_symmetric_patch(
    original_xyz: np.ndarray,
    cylinder: dict[str, Any],
) -> np.ndarray:
    pts = original_xyz.astype(np.float32).copy()
    c = np.asarray(cylinder["center_xyz"], dtype=np.float32)

    sym = pts.copy()

    # rotație 180° ca rigid body în jurul axei verticale Y
    # adică se rotește în planul XZ
    sym[:, 0] = 2.0 * c[0] - pts[:, 0]
    sym[:, 1] = pts[:, 1]
    sym[:, 2] = 2.0 * c[2] - pts[:, 2]

    return sym.astype(np.float32)

# ---------------------------------------------------------------------
# Roll / tape ring symmetry
# ---------------------------------------------------------------------

def estimate_roll_center(
    mask: np.ndarray,
    depth_m: np.ndarray,
    obj_valid: np.ndarray,
    intr: CameraIntrinsics,
) -> dict[str, Any]:
    ys, xs = np.where(obj_valid)

    if len(xs) == 0:
        raise RuntimeError("No valid object depth for roll estimation")

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    u_center = 0.5 * (x1 + x2)
    v_center = 0.5 * (y1 + y2)

    z_visible = float(np.median(depth_m[ys, xs]))
    pixel_radius = 0.5 * float(max(x2 - x1, y2 - y1))
    radius_m = pixel_radius * z_visible / float(intr.fx)
    z_center = z_visible + radius_m

    center_xyz = backproject(
        np.asarray([u_center], dtype=np.float32),
        np.asarray([v_center], dtype=np.float32),
        np.asarray([z_center], dtype=np.float32),
        intr,
    )[0].astype(np.float32)

    axis_direction = center_xyz.copy()
    axis_norm = float(np.linalg.norm(axis_direction))
    if axis_norm < 1e-6:
        axis_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        axis_direction = (axis_direction / axis_norm).astype(np.float32)

    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "center_uv": [float(u_center), float(v_center)],
        "center_xyz": center_xyz.astype(float).tolist(),
        "z_center_m": float(z_center),
        "z_visible_m": float(z_visible),
        "radius_m": float(radius_m),
        "axis_direction_xyz": axis_direction.astype(float).tolist(),
    }


def roll_symmetric_patch(
    original_xyz: np.ndarray,
    roll_model: dict[str, Any],
) -> np.ndarray:
    pts = original_xyz.astype(np.float32)
    c = np.asarray(roll_model["center_xyz"], dtype=np.float32)

    axis = np.asarray(
        roll_model.get("axis_direction_xyz", [0.0, 0.0, 1.0]),
        dtype=np.float32,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        axis = axis / axis_norm

    sym = 2.0 * c[None, :] - pts
    return sym.astype(np.float32)


# ---------------------------------------------------------------------
# Fallback 2D mirror for non-cylinder symmetric objects
# ---------------------------------------------------------------------

def mirror_pixels_vertical_mask_center(uv: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask.astype(bool))

    if len(xs) == 0:
        return uv.copy()

    u_center = 0.5 * (int(xs.min()) + int(xs.max()))

    out = uv.astype(np.float32).copy()
    out[:, 0] = 2.0 * u_center - out[:, 0]

    return np.rint(out).astype(np.int32)


def valid_mirrored_pairs(
    original_uv: np.ndarray,
    mirrored_uv: np.ndarray,
    mask: np.ndarray,
    obj_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape[:2]

    u = mirrored_uv[:, 0]
    v = mirrored_uv[:, 1]

    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    valid_pair = np.zeros(len(original_uv), dtype=bool)
    idx = np.where(inside)[0]

    if len(idx) > 0:
        u2 = u[idx]
        v2 = v[idx]
        ok = mask[v2, u2].astype(bool) & obj_valid[v2, u2].astype(bool)
        valid_pair[idx] = ok

    return original_uv[valid_pair], mirrored_uv[valid_pair]


def rotate_pixels_180_center(uv: np.ndarray, center_uv: np.ndarray) -> np.ndarray:
    out = uv.astype(np.float32).copy()
    center_uv = np.asarray(center_uv, dtype=np.float32)
    out[:, 0] = 2.0 * center_uv[0] - out[:, 0]
    out[:, 1] = 2.0 * center_uv[1] - out[:, 1]
    return np.rint(out).astype(np.int32)


# ---------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------

def reconstruct_one_object(
    *,
    object_name: str,
    image_stem: str,
    image_path: Path,
    rgb: np.ndarray,
    mask: np.ndarray,
    heatmap: np.ndarray,
    depth_m: np.ndarray,
    intr: CameraIntrinsics,
    output_dir: Path,
    heatmap_threshold: float,
    min_component_area: int,
    max_depth_m: float,
    depth_window_m: float,
    center_depth_offset_ratio: float,
    patch_source: str,
    patch_dilate_kernel: int,
    patch_dilate_iterations: int,
    patch_keep_percentile: float,
) -> None:
    mask = mask.astype(bool)
    heatmap = np.asarray(heatmap, dtype=np.float32)
    object_flexible = bool(flexible_for_name(object_name))

    obj_valid = object_depth_filter(
        mask=mask,
        depth_m=depth_m,
        max_depth_m=max_depth_m,
        depth_window_m=depth_window_m,
    )

    ov, ou = np.where(obj_valid)

    if len(ou) == 0:
        raise RuntimeError(f"No valid object points for {object_name}")

    object_xyz = backproject(ou, ov, depth_m[ov, ou], intr)
    object_rgb = rgb[ov, ou].astype(np.uint8)

    object_is_roll = is_roll_like(object_name)
    object_is_cylinder = is_cylinder_like(object_name) and not object_is_roll

    cylinder = None
    roll_model = None

    if object_is_roll:
        roll_model = estimate_roll_center(
            mask=mask,
            depth_m=depth_m,
            obj_valid=obj_valid,
            intr=intr,
        )
        model_method = "semantic_roll_center_180deg"

    elif object_is_cylinder:
        cylinder = estimate_semantic_cylinder(
            mask=mask,
            depth_m=depth_m,
            obj_valid=obj_valid,
            intr=intr,
            center_depth_offset_ratio=center_depth_offset_ratio,
        )
        model_method = "semantic_cylinder_180deg"

    else:
        model_method = "fallback_2d_mirror_backproject"

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

    obj_dir = output_dir / f"{image_stem}_{safe_name(object_name)}"
    obj_dir.mkdir(parents=True, exist_ok=True)

    write_ply(obj_dir / "object_cloud.ply", object_xyz, object_rgb)

    metadata: dict[str, Any] = {
        "object_name": object_name,
        "flexible": object_flexible,
        "object_flexible": object_flexible,
        "image_path": str(image_path),
        "method": model_method,
        "num_object_points": int(len(object_xyz)),
        "num_patches": int(len(components)),
        "heatmap_threshold": float(heatmap_threshold),
        "patch_source": patch_source,
        "patch_dilate_kernel": int(patch_dilate_kernel),
        "patch_dilate_iterations": int(patch_dilate_iterations),
        "patch_keep_percentile": float(patch_keep_percentile),
        "min_component_area": int(min_component_area),
        "depth_window_m": float(depth_window_m),
        "intrinsics": asdict(intr),
        "semantic_cylinder": cylinder,
        "roll_model": roll_model,
        "patches": [],
    }

    overlay = rgb.copy()

    overlay[mask] = (
        0.5 * overlay[mask].astype(np.float32)
        + 0.5 * np.array([80, 80, 255], dtype=np.float32)
    ).astype(np.uint8)

    overlay[patch_binary] = (
        0.5 * overlay[patch_binary].astype(np.float32)
        + 0.5 * np.array([0, 255, 0], dtype=np.float32)
    ).astype(np.uint8)

    if cylinder is not None:
        c_uv = np.asarray(cylinder["center_uv"], dtype=np.float32)
        h, _ = mask.shape[:2]

        cv2.line(
            overlay,
            (int(round(c_uv[0])), 0),
            (int(round(c_uv[0])), h - 1),
            (255, 165, 0),
            2,
        )

        cv2.circle(
            overlay,
            tuple(np.rint(c_uv).astype(int)),
            5,
            (255, 0, 255),
            -1,
        )

    if roll_model is not None:
        c_uv = np.asarray(roll_model["center_uv"], dtype=np.float32)
        cv2.circle(
            overlay,
            tuple(np.rint(c_uv).astype(int)),
            6,
            (255, 0, 255),
            -1,
        )

    cv2.imwrite(
        str(obj_dir / "debug_mask_patch_semantic_model.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )

    combined_points = [object_xyz]
    combined_colors = [np.full((len(object_xyz), 3), 170, dtype=np.uint8)]

    if cylinder is not None:
        axis_seg = np.asarray(cylinder["axis_segment_xyz"], dtype=np.float32)
        write_ply(
            obj_dir / "semantic_cylinder_axis.ply",
            axis_seg,
            np.asarray([[255, 165, 0], [255, 165, 0]], dtype=np.uint8),
        )

    for patch_id, comp in enumerate(components, start=1):
        pv, pu = np.where(comp)

        original_uv = np.stack([pu, pv], axis=1).astype(np.int32)

        u0 = original_uv[:, 0]
        v0 = original_uv[:, 1]

        original_xyz = backproject(u0, v0, depth_m[v0, u0], intr)
        heat_values = heatmap[v0, u0].astype(np.float32)
        symmetric_heat_values = heat_values

        original_xyz, heat_values, keep_mask = filter_patch_3d_by_centroid(
            original_xyz,
            heat_values,
            keep_percentile=patch_keep_percentile,
        )

        original_uv = original_uv[keep_mask]

        if object_is_roll and roll_model is not None:
            symmetric_xyz = roll_symmetric_patch(original_xyz, roll_model)
            symmetric_uv = np.full_like(original_uv, -1)
            symmetric_heat_values = heat_values
            valid_pairs = len(original_xyz)

        elif object_is_cylinder and cylinder is not None:
            symmetric_xyz = semantic_cylinder_symmetric_patch(original_xyz, cylinder)
            symmetric_uv = np.full_like(original_uv, -1)
            valid_pairs = len(original_xyz)
            symmetric_heat_values = heat_values

        else:
            mirrored_uv_all = mirror_pixels_vertical_mask_center(original_uv, mask)

            original_uv_pair, mirrored_uv_pair = valid_mirrored_pairs(
                original_uv,
                mirrored_uv_all,
                mask,
                obj_valid,
            )

            if len(original_uv_pair) == 0:
                print(f"[WARNING] {object_name} patch {patch_id:02d}: no valid mirrored pairs")
                continue

            u0 = original_uv_pair[:, 0]
            v0 = original_uv_pair[:, 1]
            u1 = mirrored_uv_pair[:, 0]
            v1 = mirrored_uv_pair[:, 1]

            original_xyz = backproject(u0, v0, depth_m[v0, u0], intr)
            symmetric_xyz = backproject(u1, v1, depth_m[v1, u1], intr)
            heat_values = heatmap[v0, u0].astype(np.float32)
            symmetric_heat_values = heat_values

            original_uv = original_uv_pair
            symmetric_uv = mirrored_uv_pair
            valid_pairs = len(original_uv_pair)

        if len(original_xyz) == 0:
            print(f"[WARNING] {object_name} patch {patch_id:02d}: empty after filtering")
            continue

        c0 = original_xyz.mean(axis=0).astype(np.float32)
        c1 = symmetric_xyz.mean(axis=0).astype(np.float32)
        pair_len = min(len(original_xyz), len(symmetric_xyz))
        pair_dist = (
            np.linalg.norm(symmetric_xyz[:pair_len] - original_xyz[:pair_len], axis=1)
            if pair_len > 0
            else np.zeros((0,), dtype=np.float32)
        )

        pid = f"patch_{patch_id:02d}"

        write_ply(
            obj_dir / f"{pid}_original.ply",
            original_xyz,
            heat_colors(heat_values, (0, 220, 0)),
        )

        write_ply(
            obj_dir / f"{pid}_symmetric.ply",
            symmetric_xyz,
            heat_colors(symmetric_heat_values, (0, 220, 255)),
        )

        np.savez_compressed(
            obj_dir / f"{pid}.npz",
            original_points_xyz=original_xyz.astype(np.float32),
            symmetric_points_xyz=symmetric_xyz.astype(np.float32),
            original_pixels_uv=original_uv.astype(np.int32),
            symmetric_pixels_uv=symmetric_uv.astype(np.int32),
            heat_values=heat_values.astype(np.float32),
            centroid_original_xyz=c0.astype(np.float32),
            centroid_symmetric_xyz=c1.astype(np.float32),
            centroid_distance_m=np.float32(np.linalg.norm(c1 - c0)),
            mean_pair_distance_m=np.float32(np.mean(pair_dist)),
            median_pair_distance_m=np.float32(np.median(pair_dist)),
            max_pair_distance_m=np.float32(np.max(pair_dist)),
            object_flexible=np.asarray(object_flexible, dtype=np.bool_),
        )

        combined_points.extend([original_xyz, symmetric_xyz])
        combined_colors.extend([
            np.full((len(original_xyz), 3), [0, 220, 0], dtype=np.uint8),
            np.full((len(symmetric_xyz), 3), [0, 220, 255], dtype=np.uint8),
        ])

        metadata["patches"].append(
            {
                "patch_id": int(patch_id),
                "num_points": int(len(original_xyz)),
                "num_valid_pairs": int(valid_pairs),
                "centroid_original_xyz": c0.astype(float).tolist(),
                "centroid_symmetric_xyz": c1.astype(float).tolist(),
                "centroid_distance_m": float(np.linalg.norm(c1 - c0)),
                "mean_pair_distance_m": float(np.mean(pair_dist)),
                "median_pair_distance_m": float(np.median(pair_dist)),
                "max_pair_distance_m": float(np.max(pair_dist)),
            }
        )

        print(
            f"     patch {patch_id:02d}: points={len(original_xyz)}, "
            f"centroid_dist={np.linalg.norm(c1 - c0):.4f} m, "
            f"mean_pair_dist={np.mean(pair_dist):.4f} m"
        )

    write_ply(
        obj_dir / "combined_object_original_symmetric.ply",
        np.vstack(combined_points),
        np.vstack(combined_colors),
    )

    (obj_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Symmetric grasp reconstruction using semantic object priors."
    )

    parser.add_argument("--heatmap-model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--depth", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_symmetric_final"))

    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--depth-unit", choices=["m", "mm"], default="m")
    parser.add_argument("--max-depth", type=float, default=3.0)
    parser.add_argument("--depth-window", type=float, default=0.08)

    parser.add_argument("--heatmap-threshold", type=float, default=0.20)
    parser.add_argument("--min-component-area", type=int, default=100)

    parser.add_argument(
        "--patch-source",
        choices=["heatmap", "mask"],
        default="heatmap",
        help="heatmap = use heatmap threshold; mask = use full object mask as patch.",
    )

    parser.add_argument(
        "--patch-dilate-kernel",
        type=int,
        default=0,
        help="Odd kernel size for 2D patch dilation. Use 0 to disable. Example: 15.",
    )

    parser.add_argument(
        "--patch-dilate-iterations",
        type=int,
        default=0,
        help="Number of dilation iterations. Use 0 to disable.",
    )

    parser.add_argument(
        "--patch-keep-percentile",
        type=float,
        default=100.0,
        help="3D centroid filter. 100 disables filtering; 85 keeps compact core.",
    )

    parser.add_argument(
        "--center-depth-offset-ratio",
        type=float,
        default=0.5,
        help="For cylinder objects: z_center = z_visible + ratio * radius. Try 0.3, 0.5, 1.0.",
    )

    parser.add_argument("--save-pipeline-images", action="store_true")

    # Detector/SAM args.
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

    # Heatmap postprocess.
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
    for p in [args.heatmap_model, args.input, args.depth, args.intrinsics]:
        if not p.exists():
            raise RuntimeError(f"Path not found: {p}")

    if not (0.0 <= args.heatmap_threshold <= 1.0):
        raise ValueError("--heatmap-threshold must be between 0 and 1")

    if not (0.0 < args.patch_keep_percentile <= 100.0):
        raise ValueError("--patch-keep-percentile must be in (0, 100]")


def main() -> None:
    args = parse_args()
    validate_args(args)

    image_size = (args.height, args.width)

    depth_m, intr = prepare_depth_and_intrinsics(
        args.depth,
        args.intrinsics,
        image_size,
        args.depth_unit,
    )

    config = PipelineConfig(
        image_size=image_size,
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

    pipeline = SymmetryAwareSamPipeline(args.heatmap_model, config)
    predictions, pipeline_results = pipeline.run(args.input, second_input_image=None)

    print("Predictions by symmetry:")
    print(predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_pipeline_images:
        pipeline_dir = args.output_dir / "pipeline_images"

        for r in pipeline_results:
            r.save(pipeline_dir)

        print(f"Saved pipeline images to: {pipeline_dir}")

    symmetric_results = [
        r for r in pipeline_results
        if symmetry_for_name(r.object_name) == "symmetric"
    ]

    print(f"Symmetric objects to reconstruct: {len(symmetric_results)}")

    reconstructed = 0

    for r in symmetric_results:
        try:
            print(f"[OBJECT] {r.object_name}")

            reconstruct_one_object(
                object_name=r.object_name,
                image_stem=r.image_path.stem,
                image_path=r.image_path,
                rgb=r.rgb_resized,
                mask=r.sam_mask,
                heatmap=r.final_heatmap,
                depth_m=depth_m,
                intr=intr,
                output_dir=args.output_dir,
                heatmap_threshold=args.heatmap_threshold,
                min_component_area=args.min_component_area,
                max_depth_m=args.max_depth,
                depth_window_m=args.depth_window,
                center_depth_offset_ratio=args.center_depth_offset_ratio,
                patch_source=args.patch_source,
                patch_dilate_kernel=args.patch_dilate_kernel,
                patch_dilate_iterations=args.patch_dilate_iterations,
                patch_keep_percentile=args.patch_keep_percentile,
            )

            reconstructed += 1

        except Exception as exc:
            print(f"[WARNING] Failed reconstruction for {r.object_name!r}: {exc}")

    print(f"Done. Reconstructed {reconstructed}/{len(symmetric_results)} symmetric object(s).")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
