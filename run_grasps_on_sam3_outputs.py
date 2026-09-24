#!/usr/bin/env python3
"""
Adapt the new SAM3 outputs to the simpler grasp_work layout, then run grasping.

Supported upstream sources:
  1. symmetric_semantic_3d/<object>/...          (best for symmetric objects)
  2. asymmetric_twoview_3d/fused_asymmetric_objects/<object>/...
                                                 (best for asymmetric objects)
  3. root-level *_object_cloud + *_patch_*.npz   (fallback)

Why this script exists:
  The upstream pipeline now saves richer 3D outputs than grasp_work was
  originally written for. Instead of rewriting the whole grasp pipeline,
  we translate each object to the local layout grasp_work already knows:

      refactored_sam_pipeline-3/grasp_ready_objects/<object>/
          <object>_object_cloud.ply
          <object>_grasp_patch_cloud.ply
          <object>_grasp_patch_data.npz

  The npz also stores per-point 3D heat values, so downstream stages can
  score fused asymmetric patches even when no single 2D heatmap image is
  the "right" one anymore.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from object_dimensions import get_object_dimensions
def read_ascii_ply(path: Path) -> np.ndarray:
    """Read xyz from a SAM3-style ASCII PLY (x y z r g b per line)."""
    with path.open("r") as f:
        lines = f.readlines()
    end_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "end_header") + 1
    vertices = []
    for ln in lines[end_idx:]:
        parts = ln.split()
        if len(parts) < 3:
            continue
        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(vertices, dtype=np.float64)


def estimate_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
    """Simple local-PCA normals, flipped toward the origin by default."""
    n = len(points)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if n < 3:
        normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (n, 1))
        return normals

    normals = np.zeros_like(points)
    chunk = 256
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = points[start:end]
        diffs = block[:, None, :] - points[None, :, :]
        d2 = np.einsum("bij,bij->bi", diffs, diffs)
        kth = max(2, min(k, n - 1))
        idx = np.argpartition(d2, kth, axis=1)[:, :kth]
        for i, neigh in enumerate(idx):
            local = points[neigh].copy()
            local -= local.mean(axis=0)
            cov = (local.T @ local) / max(len(local) - 1, 1)
            _, eigvecs = np.linalg.eigh(cov)
            normals[start + i] = eigvecs[:, 0]

    sign = np.sign(np.einsum("ij,ij->i", normals, -points))
    sign[sign == 0] = 1.0
    normals *= sign[:, None]
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm[norm < 1e-9] = 1.0
    return normals / norm


def write_binary_ply(path: Path, points: np.ndarray, normals: np.ndarray) -> None:
    """Write binary PLY in the format grasp_work expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "property double nx\n"
        "property double ny\n"
        "property double nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype([
        ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
        ("nx", "<f8"), ("ny", "<f8"), ("nz", "<f8"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    arr = np.empty(n, dtype=dtype)
    arr["x"] = points[:, 0]
    arr["y"] = points[:, 1]
    arr["z"] = points[:, 2]
    arr["nx"] = normals[:, 0]
    arr["ny"] = normals[:, 1]
    arr["nz"] = normals[:, 2]
    arr["red"] = 200
    arr["green"] = 200
    arr["blue"] = 200
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        arr.tofile(f)


def stem_to_prefix(name: str) -> str:
    """
    Convert upstream labels to our local prefix style.

    Examples:
      rgb_1778075546_blue_cup  -> blue_cup
      roll of tape             -> roll_of_tape
      phillips screwdriver     -> phillips_screwdriver
    """
    cleaned = name.replace(" ", "_").replace("-", "_")
    parts = [p for p in cleaned.split("_") if p]
    label_parts = [p for p in parts if p.lower() != "rgb" and not p.isdigit()]
    return "_".join(label_parts) or cleaned.lower()


def stack_arrays(arrays: list[np.ndarray], width: int, dtype) -> np.ndarray:
    """Concatenate arrays or return an empty array with the requested width."""
    valid = [a for a in arrays if len(a) > 0]
    if not valid:
        return np.zeros((0, width), dtype=dtype)
    return np.concatenate(valid, axis=0).astype(dtype, copy=False)


def transfer_sparse_point_values_nn(
    target_points: np.ndarray,
    source_points: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    """
    Transfer per-point values by nearest-neighbor only, without distance decay.

    For asymmetric per-view comparisons the raw patch and aligned patch may be
    far apart numerically even though they describe the same observed surface in
    two nearby frames of the same reconstruction process. A decayed transfer
    can collapse the heat to ~0 everywhere, so here we preserve the upstream
    heat magnitude and only remap by nearest point correspondence.
    """
    if len(target_points) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(source_points) == 0 or len(source_values) != len(source_points):
        return np.zeros((len(target_points),), dtype=np.float64)

    out = np.zeros((len(target_points),), dtype=np.float64)
    chunk = 256
    for start in range(0, len(target_points), chunk):
        end = min(start + chunk, len(target_points))
        block = target_points[start:end]
        diffs = block[:, None, :] - source_points[None, :, :]
        d2 = np.einsum("bij,bij->bi", diffs, diffs)
        nn_idx = np.argmin(d2, axis=1)
        out[start:end] = source_values[nn_idx]
    return np.clip(out, 0.0, 1.0)


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_PIPELINE_ROOT = PROJECT_ROOT / "refactored_sam_pipeline-3"
PIPELINE_ROOT = Path(
    os.environ.get("SAM_PIPELINE_ROOT", str(DEFAULT_PIPELINE_ROOT))
).expanduser().resolve()
OUTPUT_ROOT = Path(
    os.environ.get("GRASP_OUTPUT_ROOT", str(DEFAULT_PIPELINE_ROOT))
).expanduser().resolve()
DATA_DIR = PIPELINE_ROOT / "data"
GRASP_OBJ_DIR = OUTPUT_ROOT / "grasp_ready_objects"
HEATMAP_DIR = OUTPUT_ROOT / "grasp_ready_heatmaps"
CONSISTENCY_REPORT_PATH = OUTPUT_ROOT / "grasp_consistency_report.json"
def flexible_for_source_name(name: str, metadata: dict | None = None) -> bool:
    """Resolve object flexibility from upstream metadata, falling back to the catalog."""
    if metadata is not None:
        for key in ("object_flexible", "flexible"):
            if key in metadata:
                return bool(metadata[key])

    try:
        pipeline_root_str = str(PIPELINE_ROOT)
        if pipeline_root_str not in sys.path:
            sys.path.insert(0, pipeline_root_str)
        from src.object_catalog import flexible_for_name as _catalog_flexible_for_name

        return bool(_catalog_flexible_for_name(name))
    except Exception:
        return False


def pipeline_output_dirs() -> list[Path]:
    """List plausible SAM output folders under the active pipeline root."""
    candidates: list[Path] = []
    for path in sorted(PIPELINE_ROOT.glob("outputs*")):
        if not path.is_dir():
            continue
        if (
            (path / "symmetric_semantic_3d").exists()
            or (path / "asymmetric_twoview_3d").exists()
            or any(path.glob("*_heatmap_cloud.npz"))
            or any((path / "heatmap_pointclouds").glob("*.ply"))
        ):
            candidates.append(path)
    return candidates


def resolve_sam_outputs_dir() -> Path:
    """
    Pick the current SAM pipeline output folder.

    Prefer the canonical retry folder when it exists. Otherwise, score all
    plausible `outputs*` folders and keep the richest one.
    """
    def candidate_score(path: Path) -> tuple[int, float]:
        sym = sum(1 for p in (path / "symmetric_semantic_3d").iterdir()) if (path / "symmetric_semantic_3d").exists() else 0
        asym_root = path / "asymmetric_twoview_3d" / "fused_asymmetric_objects"
        asym = sum(1 for p in asym_root.iterdir()) if asym_root.exists() else 0
        pointclouds = sum(1 for _ in path.glob("heatmap_pointclouds/*.ply"))
        root_npz = sum(1 for _ in path.glob("*_heatmap_cloud.npz"))
        score = (sym * 1000) + (asym * 1000) + (pointclouds * 10) + root_npz
        return score, path.stat().st_mtime

    preferred = PIPELINE_ROOT / "outputs_full_pipeline_gpu_retry"
    if preferred.exists():
        return preferred
    candidates = pipeline_output_dirs()
    if candidates:
        return max(candidates, key=candidate_score)
    return preferred


SAM_OUTPUTS = Path(
    os.environ.get("SAM_PIPELINE_OUTPUTS", str(resolve_sam_outputs_dir()))
).expanduser().resolve()


def iter_available_output_dirs() -> list[Path]:
    """List candidate pipeline output folders, keeping the active one first."""
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in [SAM_OUTPUTS, *pipeline_output_dirs()]:
        if not path.is_dir():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(path)
    return ordered


def fused_patch_extent_stats(obj_dir: Path) -> tuple[np.ndarray, int]:
    """Measure the fused asymmetric patch size for one upstream object dir."""
    npz_path = obj_dir / "fused_data.npz"
    if npz_path.exists():
        npz = np.load(npz_path, allow_pickle=True)
        if "fused_patch_world_xyz" in npz.files:
            points = np.asarray(npz["fused_patch_world_xyz"], dtype=np.float64)
            if len(points) > 0:
                return points.max(axis=0) - points.min(axis=0), int(len(points))

    ply_path = obj_dir / "combined_aligned_patches.ply"
    if ply_path.exists():
        points = read_ascii_ply(ply_path)
        if len(points) > 0:
            return points.max(axis=0) - points.min(axis=0), int(len(points))

    return np.array([np.inf, np.inf, np.inf], dtype=np.float64), 0


def asymmetric_extent_score(prefix: str, extents_m: np.ndarray, point_count: int) -> float:
    """
    Score one asymmetric fused reconstruction candidate.

    Lower is better. We strongly penalize reconstructions whose size is far
    outside the expected handheld-object scale, while lightly preferring
    denser point clouds when two candidates are similarly plausible.
    """
    extents_m = np.asarray(extents_m, dtype=np.float64)
    if point_count <= 0 or not np.all(np.isfinite(extents_m)):
        return float("inf")

    diag_m = float(np.linalg.norm(extents_m))
    prior = get_object_dimensions(prefix)
    if prior is None:
        return diag_m - (1e-6 * point_count)

    ref_extents = np.sort(np.asarray(prior.bounding_box_xyz_m, dtype=np.float64))
    cand_extents = np.sort(np.maximum(extents_m, 1e-6))
    log_ratio = np.abs(np.log(cand_extents / np.maximum(ref_extents, 1e-6)))
    size_error = float(np.mean(log_ratio))

    ref_diag_m = float(np.linalg.norm(ref_extents))
    oversize_ratio = max(
        float(np.max(extents_m) / max(float(np.max(ref_extents)), 1e-6)),
        diag_m / max(ref_diag_m, 1e-6),
    )
    oversize_penalty = max(0.0, oversize_ratio - 2.5) * 4.0
    density_bonus = -1e-6 * point_count
    return size_error + oversize_penalty + density_bonus


def resolve_best_asymmetric_object_dir(obj_dir: Path, prefix: str) -> Path:
    """
    Pick the most plausible fused asymmetric reconstruction across runs.

    Some reruns produced stretched fused reconstructions for a subset of
    objects. When multiple `outputs_full_pipeline*` folders are available,
    we keep the best-scoring one for the same object name instead of blindly
    trusting the newest folder.
    """
    candidates: list[tuple[float, float, int, Path]] = []
    for outputs_dir in iter_available_output_dirs():
        candidate_dir = (
            outputs_dir
            / "asymmetric_twoview_3d"
            / "fused_asymmetric_objects"
            / obj_dir.name
        )
        if not candidate_dir.exists():
            continue
        extents_m, point_count = fused_patch_extent_stats(candidate_dir)
        score = asymmetric_extent_score(prefix, extents_m, point_count)
        diag_m = float(np.linalg.norm(extents_m)) if np.all(np.isfinite(extents_m)) else float("inf")
        candidates.append((score, diag_m, -point_count, candidate_dir))

    if not candidates:
        return obj_dir
    candidates.sort(key=lambda item: (item[0], item[1], item[2], str(item[3])))
    return candidates[0][3]


def normalize_vector(
    vector: np.ndarray,
    fallback: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return vector / norm


def rotate_180_about_axis(
    points_xyz: np.ndarray,
    center_xyz: np.ndarray,
    axis_direction_xyz: np.ndarray,
) -> np.ndarray:
    """Mirror points by a 180 degree rotation around the estimated roll axis."""
    pts = np.asarray(points_xyz, dtype=np.float64)
    center_xyz = np.asarray(center_xyz, dtype=np.float64)
    axis_direction_xyz = normalize_vector(axis_direction_xyz)

    centered = pts - center_xyz[None, :]
    axis_component = np.sum(
        centered * axis_direction_xyz[None, :], axis=1, keepdims=True,
    )
    axis_component = axis_component * axis_direction_xyz[None, :]
    return center_xyz[None, :] + (2.0 * axis_component - centered)


def build_roll_symmetric_object_cloud(obj_dir: Path) -> np.ndarray:
    """
    Rebuild roll/tape symmetry in-memory from the raw visible object cloud.

    Some saved roll reconstructions were produced with a centre reflection
    instead of a 180 degree rotation around the roll axis, which visibly
    offsets the mirrored half. For grasping we correct only the derived full
    object cloud, while keeping the original upstream patch heat untouched.
    """
    object_xyz = read_ascii_ply(obj_dir / "object_cloud.ply")
    if len(object_xyz) == 0:
        return object_xyz

    center_xyz = object_xyz.mean(axis=0)
    centered = object_xyz - center_xyz[None, :]
    if len(object_xyz) >= 3:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis_direction_xyz = normalize_vector(vt[-1])
    else:
        axis_direction_xyz = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    combined_parts = [object_xyz]
    for npz_path in sorted(obj_dir.glob("patch_*.npz")):
        npz = np.load(npz_path, allow_pickle=True)
        original_xyz = np.asarray(npz["original_points_xyz"], dtype=np.float64)
        symmetric_xyz = rotate_180_about_axis(
            original_xyz, center_xyz, axis_direction_xyz,
        )
        combined_parts.extend([original_xyz, symmetric_xyz])

    return stack_arrays(combined_parts, 3, np.float64)


def build_semantic_cylinder_object_cloud(
    obj_dir: Path,
    metadata: dict,
) -> np.ndarray:
    """
    Densify a semantic cylindrical reconstruction into a full 360° shell.

    The upstream symmetric output already estimates a cylinder axis and radius
    from the mask/depth pair. The stored point clouds are good for debugging
    but often too sparse to look like a completed object in the no-heatmap
    baseline. Here we keep the original upstream points and add a denser shell
    sampled from the semantic cylinder parameters.
    """
    combined_object = obj_dir / "combined_object_original_symmetric.ply"
    base_cloud = combined_object if combined_object.exists() else obj_dir / "object_cloud.ply"
    base_points = read_ascii_ply(base_cloud) if base_cloud.exists() else np.zeros((0, 3), dtype=np.float64)

    cyl = metadata.get("semantic_cylinder")
    if not cyl:
        return base_points

    axis_segment = np.asarray(cyl.get("axis_segment_xyz", []), dtype=np.float64)
    if axis_segment.shape != (2, 3):
        return base_points

    axis_start = axis_segment[0]
    axis_end = axis_segment[1]
    axis_vec = axis_end - axis_start
    axis_len = float(np.linalg.norm(axis_vec))
    if axis_len < 1e-6:
        return base_points
    axis_dir = normalize_vector(axis_vec)

    radius_m = float(cyl.get("radius_m", 0.0))
    if radius_m < 1e-4:
        return base_points

    trial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(trial @ axis_dir)) > 0.9:
        trial = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    tangent_u = normalize_vector(np.cross(axis_dir, trial))
    tangent_v = normalize_vector(np.cross(axis_dir, tangent_u))

    circumference_m = 2.0 * np.pi * radius_m
    theta_count = int(np.clip(np.ceil(circumference_m / 0.004), 48, 128))
    axis_count = int(np.clip(np.ceil(axis_len / 0.004), 24, 96))

    theta = np.linspace(0.0, 2.0 * np.pi, theta_count, endpoint=False)
    axis_t = np.linspace(0.0, axis_len, axis_count)
    ring_dirs = (
        np.cos(theta)[:, None] * tangent_u[None, :]
        + np.sin(theta)[:, None] * tangent_v[None, :]
    )

    shell_parts = []
    for t in axis_t:
        center = axis_start + t * axis_dir
        shell_parts.append(center[None, :] + radius_m * ring_dirs)

    top_rim = axis_end[None, :] + radius_m * ring_dirs
    bottom_rim = axis_start[None, :] + radius_m * ring_dirs
    dense_shell = stack_arrays(shell_parts + [top_rim, bottom_rim], 3, np.float64)
    return stack_arrays([base_points, dense_shell], 3, np.float64)


def load_intrinsics() -> tuple[float, float, float, float]:
    data = json.loads((DATA_DIR / "intrinsics.json").read_text())
    return float(data["fx"]), float(data["fy"]), float(data["cx"]), float(data["cy"])


def load_depth_points_camera(
    image_ref: str,
    step: int = 6,
    max_depth_m: float = 3.0,
) -> np.ndarray:
    """Backproject one RGB-associated depth map into camera-frame scene points."""
    image_name = Path(image_ref).name
    frame_id = image_name.replace("rgb_", "").replace(".png", "")
    depth_path = DATA_DIR / f"depth_{frame_id}.npy"
    if not depth_path.exists():
        return np.zeros((0, 3), dtype=np.float64)

    depth_m = np.asarray(np.load(depth_path), dtype=np.float32)
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]

    fx, fy, cx, cy = load_intrinsics()
    height, width = depth_m.shape[:2]
    ys = np.arange(0, height, step, dtype=np.int32)
    xs = np.arange(0, width, step, dtype=np.int32)
    uu, vv = np.meshgrid(xs, ys)
    z = depth_m[vv, uu]
    valid = np.isfinite(z) & (z > 0.0) & (z <= max_depth_m)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)

    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.column_stack((x, y, z))


def transform_points(points: np.ndarray, pose_cam_to_world: np.ndarray | None) -> np.ndarray:
    """Apply a 4x4 camera-to-world transform when needed."""
    if pose_cam_to_world is None or len(points) == 0:
        return points
    homog = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    world = (pose_cam_to_world @ homog.T).T
    return world[:, :3]

# Grasping mode:
#   primary_only  -> one grasp run per object, using the best available source
#                    produced by the current SAM pipeline run
#   compare_views -> old behavior: primary result + comparison variants
GRASP_EVAL_MODE = os.environ.get("GRASP_EVAL_MODE", "primary_only").strip().lower()
if GRASP_EVAL_MODE not in {"primary_only", "compare_views"}:
    GRASP_EVAL_MODE = "primary_only"

PRIMARY_SOURCE_PRIORITY = (
    "symmetric_semantic",
    "asymmetric_fused",
    "root_single_view",
    "root_heatmap_npz",
    "root_heatmap_pointcloud",
    "view2_heatmap_pointcloud",
)
PRIMARY_ONLY_SOURCE_KINDS = ("symmetric_semantic", "asymmetric_fused")

CANONICAL_PREFIX = {
    "black_computer_mouse": "computer_mouse",
    "white_computer_mouse": "computer_mouse",
    "phillips_screwdriver": "screwdriver",
    "philips_screwdriver": "screwdriver",
}


def canonical_source_prefix(prefix: str) -> str:
    """Merge prompt-variant labels that refer to the same physical object."""
    return CANONICAL_PREFIX.get(prefix, prefix)


def _frame_id_from_stem(stem: str) -> str:
    parts = stem.split("_")
    for p in parts:
        if p.isdigit():
            return p
    return stem


def _safe_tag(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def collect_available_sources() -> dict[str, list[dict]]:
    """
    Gather ALL usable upstream sources, without collapsing them to one choice.

    This lets us follow the evaluation plan explicitly:
      - symmetric: primary = symmetric reconstruction, compare with other views
      - asymmetric: primary = fused two-view reconstruction, compare with
        available component views
    """
    by_prefix: dict[str, list[dict]] = {}

    def add(src: dict) -> None:
        by_prefix.setdefault(src["prefix"], []).append(src)

    sym_root = SAM_OUTPUTS / "symmetric_semantic_3d"
    if sym_root.exists():
        for obj_dir in sorted(p for p in sym_root.iterdir() if p.is_dir()):
            prefix = canonical_source_prefix(stem_to_prefix(obj_dir.name))
            metadata = json.loads((obj_dir / "metadata.json").read_text()) if (obj_dir / "metadata.json").exists() else {}
            add({
                "kind": "symmetric_semantic",
                "prefix": prefix,
                "path": obj_dir,
                "object_flexible": flexible_for_source_name(prefix, metadata),
                "source_label": "primary_symmetric_3d",
                "run_tag": "primary_symmetric_3d",
                "object_class": "symmetric",
                "is_primary": True,
            })

    asym_root = SAM_OUTPUTS / "asymmetric_twoview_3d" / "fused_asymmetric_objects"
    if asym_root.exists():
        for obj_dir in sorted(p for p in asym_root.iterdir() if p.is_dir()):
            prefix = canonical_source_prefix(stem_to_prefix(obj_dir.name))
            resolved_obj_dir = resolve_best_asymmetric_object_dir(obj_dir, prefix)
            meta = json.loads((resolved_obj_dir / "metadata.json").read_text())
            add({
                "kind": "asymmetric_fused",
                "prefix": prefix,
                "path": resolved_obj_dir,
                "object_flexible": flexible_for_source_name(prefix, meta),
                "source_label": "primary_asymmetric_fused",
                "run_tag": "primary_asymmetric_fused",
                "object_class": "asymmetric",
                "is_primary": True,
            })
            for view_key in (1, 2):
                image_ref = meta.get(f"view{view_key}_image", "")
                frame_id = _frame_id_from_stem(Path(image_ref).stem)
                add({
                    "kind": "asymmetric_view_patch",
                    "prefix": prefix,
                    "path": resolved_obj_dir,
                    "object_flexible": flexible_for_source_name(prefix, meta),
                    "view_key": view_key,
                    "frame_id": frame_id,
                    "source_label": f"comparison_view{view_key}_{frame_id}",
                    "run_tag": f"comparison_view{view_key}_{frame_id}",
                    "object_class": "asymmetric",
                    "is_primary": False,
                })

    # Direct-view sources for comparison on symmetric objects, and as a
    # generic fallback when no class-specific primary source exists.
    seen_direct: set[tuple[str, str]] = set()
    for npz_path in sorted(SAM_OUTPUTS.glob("*_heatmap_cloud.npz")):
        stem = npz_path.name.replace("_heatmap_cloud.npz", "")
        prefix = canonical_source_prefix(stem_to_prefix(stem))
        frame_id = _frame_id_from_stem(stem)
        key = (prefix, frame_id)
        add({
            "kind": "root_heatmap_npz",
            "prefix": prefix,
            "object_flexible": flexible_for_source_name(prefix),
            "path": npz_path,
            "stem": stem,
            "frame_id": frame_id,
            "source_label": f"comparison_view_{frame_id}",
            "run_tag": f"comparison_view_{frame_id}",
            "object_class": "direct_view",
            "is_primary": False,
        })
        seen_direct.add(key)

    for ply_path in sorted((SAM_OUTPUTS / "heatmap_pointclouds").glob("*_heatmap_pointcloud.ply")):
        stem = ply_path.name.replace("_heatmap_pointcloud.ply", "")
        prefix = canonical_source_prefix(stem_to_prefix(stem))
        frame_id = _frame_id_from_stem(stem)
        key = (prefix, frame_id)
        if key in seen_direct:
            continue
        add({
            "kind": "root_heatmap_pointcloud",
            "prefix": prefix,
            "object_flexible": flexible_for_source_name(prefix),
            "path": ply_path,
            "stem": stem,
            "frame_id": frame_id,
            "source_label": f"comparison_view_{frame_id}",
            "run_tag": f"comparison_view_{frame_id}",
            "object_class": "direct_view",
            "is_primary": False,
        })
        seen_direct.add(key)

    view2_root = SAM_OUTPUTS / "asymmetric_view2_heatmaps" / "heatmap_pointclouds"
    for ply_path in sorted(view2_root.glob("*_heatmap_pointcloud.ply")):
        stem = ply_path.name.replace("_heatmap_pointcloud.ply", "")
        prefix = canonical_source_prefix(stem_to_prefix(stem))
        frame_id = _frame_id_from_stem(stem)
        key = (prefix, frame_id)
        if key in seen_direct:
            continue
        add({
            "kind": "view2_heatmap_pointcloud",
            "prefix": prefix,
            "object_flexible": flexible_for_source_name(prefix),
            "path": ply_path,
            "stem": stem,
            "frame_id": frame_id,
            "source_label": f"comparison_view_{frame_id}",
            "run_tag": f"comparison_view_{frame_id}",
            "object_class": "direct_view",
            "is_primary": False,
        })
        seen_direct.add(key)

    # Fallback only if a prefix has no class-specific primary.
    for ply in sorted(SAM_OUTPUTS.glob("*_object_cloud.ply")):
        stem = ply.name.replace("_object_cloud.ply", "")
        prefix = canonical_source_prefix(stem_to_prefix(stem))
        if not list(SAM_OUTPUTS.glob(f"{stem}_patch_*_original.npz")):
            continue
        add({
            "kind": "root_single_view",
            "prefix": prefix,
            "object_flexible": flexible_for_source_name(prefix),
            "path": SAM_OUTPUTS,
            "stem": stem,
            "frame_id": _frame_id_from_stem(stem),
            "source_label": "single_view_fallback",
            "run_tag": f"single_view_fallback_{_frame_id_from_stem(stem)}",
            "object_class": "direct_view",
            "is_primary": False,
        })

    return by_prefix


def choose_primary_source(entries: list[dict]) -> dict | None:
    """Pick one upstream source per object for direct grasping."""
    for kind in PRIMARY_SOURCE_PRIORITY:
        src = next((s for s in entries if s["kind"] == kind), None)
        if src is not None:
            return src
    return None


def infer_object_class(source: dict) -> str:
    """Normalize the source kind into the coarse class used in reports."""
    if source["kind"] == "symmetric_semantic":
        return "symmetric"
    if source["kind"] == "asymmetric_fused":
        return "asymmetric"
    return "direct_view"


def build_object_evaluations(source_index: dict[str, list[dict]]) -> list[dict]:
    """Build one evaluation job per object from the current SAM outputs."""
    evaluations: list[dict] = []
    for prefix in sorted(source_index):
        entries = source_index[prefix]
        if GRASP_EVAL_MODE != "compare_views":
            primaries = [s for s in entries if s["kind"] in PRIMARY_ONLY_SOURCE_KINDS]
            primaries.sort(key=lambda s: PRIMARY_ONLY_SOURCE_KINDS.index(s["kind"]))
            for primary in primaries:
                evaluations.append({
                    "prefix": prefix,
                    "object_class": infer_object_class(primary),
                    "primary": primary,
                    "comparisons": [],
                })
            continue

        primary = choose_primary_source(entries)
        if primary is None:
            continue
        object_class = infer_object_class(primary)

        if object_class == "symmetric":
            comparisons = [
                s for s in entries
                if s["kind"] in {"root_heatmap_npz", "root_heatmap_pointcloud", "view2_heatmap_pointcloud"}
            ]
        elif object_class == "asymmetric":
            # For asymmetric objects, prefer the component views saved inside
            # the new two-view fusion output. Those keep the per-point 3D heat
            # values aligned with the actual view geometry, so downstream
            # grasping and visualization stay heat-aware. The older direct
            # heatmap pointcloud exports remain only as a fallback.
            comparisons = [
                s for s in entries
                if s["kind"] == "asymmetric_view_patch"
            ]
            if not comparisons:
                comparisons = [
                    s for s in entries
                    if s["kind"] in {"root_heatmap_npz", "root_heatmap_pointcloud", "view2_heatmap_pointcloud"}
                ]
        else:
            comparisons = []

        comparisons.sort(key=lambda s: s["source_label"])
        evaluations.append({
            "prefix": prefix,
            "object_class": object_class,
            "primary": primary,
            "comparisons": comparisons,
        })
    return evaluations


def load_root_heatmap_npz(source: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path | None]:
    """
    Load a direct root-level heatmap cloud.

    This is the cleanest "use upstream as-is" source:
      - object_cloud.ply = observed object surface
      - heatmap_cloud.npz = same 3D points + heat values
      - pred_final_gray.png = paired 2D heatmap image
    """
    stem = source["stem"]
    npz = np.load(Path(source["path"]), allow_pickle=True)
    patch_points = np.asarray(npz["points_xyz"], dtype=np.float64)
    patch_heat = np.asarray(npz["heat_values"], dtype=np.float64)
    camera_origin = np.asarray(
        npz["camera_origin_xyz"], dtype=np.float64,
    ) if "camera_origin_xyz" in npz.files else np.zeros(3, dtype=np.float64)

    object_cloud_path = SAM_OUTPUTS / f"{stem}_object_cloud.ply"
    object_points = (
        read_ascii_ply(object_cloud_path)
        if object_cloud_path.exists()
        else patch_points.copy()
    )
    scene_points = load_depth_points_camera(f"{stem}.png")
    if len(scene_points) == 0:
        scene_points = object_points
    patch_pixels = np.zeros((len(patch_points), 2), dtype=np.int32)
    heatmap_path = SAM_OUTPUTS / f"{stem}_pred_final_gray.png"
    return object_points, scene_points, patch_points, patch_pixels, patch_heat, camera_origin, heatmap_path


def load_heatmap_pointcloud(source: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path | None]:
    """
    Load a direct exported heatmap point cloud.

    These files are already exactly the visible 3D points selected by the
    upstream pipeline. We use the same point cloud both as object cloud and as
    grasp search space, and recover the heat values later from the paired 2D
    heatmap image by projection.
    """
    stem = source["stem"]
    points = read_ascii_ply(Path(source["path"]))
    patch_pixels = np.zeros((len(points), 2), dtype=np.int32)
    patch_heat = np.zeros(0, dtype=np.float64)

    # By default the direct exported clouds are still in the main camera frame,
    # so camera origin stays at (0,0,0).
    camera_origin = np.zeros(3, dtype=np.float64)
    scene_points = load_depth_points_camera(f"{stem}.png")
    if len(scene_points) == 0:
        scene_points = points
    heatmap_path = SAM_OUTPUTS / f"{stem}_pred_final_gray.png"
    if not heatmap_path.exists():
        heatmap_path = (
            SAM_OUTPUTS / "asymmetric_view2_heatmaps" / f"{stem}_pred_final_gray.png"
        )
    return points, scene_points, points.copy(), patch_pixels, patch_heat, camera_origin, heatmap_path


def load_root_single_view(source: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path | None]:
    """Load a root-level object cloud + union of all old-style original patches."""
    stem = source["stem"]
    object_points = read_ascii_ply(SAM_OUTPUTS / f"{stem}_object_cloud.ply")

    patch_points = []
    patch_pixels = []
    patch_heat = []
    for npz_path in sorted(SAM_OUTPUTS.glob(f"{stem}_patch_*_original.npz")):
        npz = np.load(npz_path, allow_pickle=True)
        patch_points.append(np.asarray(npz["points_xyz"], dtype=np.float64))
        patch_pixels.append(np.asarray(npz["pixels_uv"], dtype=np.int32))
        patch_heat.append(np.asarray(npz["heat_values"], dtype=np.float64))

    scene_points = load_depth_points_camera(f"{stem}.png")
    if len(scene_points) == 0:
        scene_points = object_points

    return (
        object_points,
        scene_points,
        stack_arrays(patch_points, 3, np.float64),
        stack_arrays(patch_pixels, 2, np.int32),
        np.concatenate(patch_heat).astype(np.float64) if patch_heat else np.zeros(0, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        SAM_OUTPUTS / f"{stem}_pred_final_gray.png",
    )


def load_symmetric_semantic(source: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path | None]:
    """
    Load a semantic symmetric reconstruction.

    Important boundary:
      We treat the upstream result as read-only and do not invent extra heat
      values. The patch heat from SAM3 belongs to the ORIGINAL visible points,
      so for grasp-zone search we keep exactly those points and their heat.
      The full object cloud can still be the semantic symmetric object cloud
      for collision/context, because that is also an upstream output as-is.
    """
    obj_dir = Path(source["path"])
    metadata = json.loads((obj_dir / "metadata.json").read_text())
    if metadata.get("roll_model") and not metadata.get("semantic_cylinder"):
        object_points = build_roll_symmetric_object_cloud(obj_dir)
    else:
        combined_object = obj_dir / "combined_object_original_symmetric.ply"
        object_cloud = combined_object if combined_object.exists() else obj_dir / "object_cloud.ply"
        object_points = read_ascii_ply(object_cloud)
    image_ref = metadata.get("image_path", "")
    scene_points = load_depth_points_camera(image_ref)
    if len(scene_points) == 0:
        scene_points = object_points

    patch_points = []
    patch_pixels = []
    patch_heat = []
    for npz_path in sorted(obj_dir.glob("patch_*.npz")):
        npz = np.load(npz_path, allow_pickle=True)
        patch_points.append(np.asarray(npz["original_points_xyz"], dtype=np.float64))
        patch_pixels.append(np.asarray(npz["original_pixels_uv"], dtype=np.int32))
        patch_heat.append(np.asarray(npz["heat_values"], dtype=np.float64))

    return (
        object_points,
        scene_points,
        stack_arrays(patch_points, 3, np.float64),
        stack_arrays(patch_pixels, 2, np.int32),
        np.concatenate(patch_heat).astype(np.float64) if patch_heat else np.zeros(0, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        None,
    )


def load_asymmetric_fused(source: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path | None]:
    """
    Load a fused two-view asymmetric reconstruction.

    The fused patch already lives in a common 3D frame. For grasp generation we
    still need one observing origin to define the visible side / approach
    direction. By default we use view 1, so the fused object is still grasped
    "from the main camera direction" rather than from a synthetic midpoint.
    """
    obj_dir = Path(source["path"])
    npz = np.load(obj_dir / "fused_data.npz", allow_pickle=True)
    meta = json.loads((obj_dir / "metadata.json").read_text())

    object_cloud_path = obj_dir / "combined_aligned_patches.ply"
    if not object_cloud_path.exists():
        object_cloud_path = obj_dir / "view2_object_world_raw.ply"
    object_points = read_ascii_ply(object_cloud_path)

    patch_points = np.asarray(npz["fused_patch_world_xyz"], dtype=np.float64)
    patch_pixels = np.full((len(patch_points), 2), -1, dtype=np.int32)
    patch_heat = np.asarray(npz["fused_heat_values"], dtype=np.float64)

    pose1 = np.asarray(meta["pose1_cam_to_world"], dtype=np.float64)
    pose2 = np.asarray(meta["pose2_cam_to_world"], dtype=np.float64)
    # Fused data is stored in view1's camera frame, so camera1 is at the origin
    # of that frame. Using pose1[:3,3] (world translation) puts the camera at
    # the same location as the object (6mm apart) because the world frame is
    # defined relative to an external robot/scene origin, not the object.
    camera_origin = np.zeros(3, dtype=np.float64)
    scene1 = transform_points(load_depth_points_camera(meta["view1_image"]), pose1)
    scene2 = transform_points(load_depth_points_camera(meta["view2_image"]), pose2)
    scene_points = stack_arrays([scene1, scene2], 3, np.float64)
    if len(scene_points) == 0:
        scene_points = object_points

    return object_points, scene_points, patch_points, patch_pixels, patch_heat, camera_origin, None


def load_asymmetric_view_patch(source: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path | None]:
    """
    Load one component view from the asymmetric two-view reconstruction.

    This lets us compare the final fused grasp with the grasps obtained from
    the individual observed views, without mixing those views into the main
    grasp selection.
    """
    obj_dir = Path(source["path"])
    npz = np.load(obj_dir / "fused_data.npz", allow_pickle=True)
    meta = json.loads((obj_dir / "metadata.json").read_text())
    view_key = int(source["view_key"])

    # For per-view comparison we want the object and the contact-search points
    # in the SAME frame. The upstream asymmetric fusion stores both:
    #   - raw per-view object points / raw patch points
    #   - an aligned patch used later by the two-view fusion
    #
    # The aligned patch is useful for fusion, but it may no longer overlap
    # the standalone per-view object cloud. If we mix those two, the local
    # heat prior gets projected onto the wrong place and the visualization
    # shows "floating" or all-blue points. For the single-view comparison we
    # therefore keep the RAW per-view geometry as the object/search space and
    # only transfer the filtered heat values back onto that raw cloud.
    object_points = np.asarray(
        npz[f"view{view_key}_object_world_xyz"], dtype=np.float64,
    )
    raw_patch_points = np.asarray(
        npz.get(f"view{view_key}_patch_world_raw_xyz", object_points),
        dtype=np.float64,
    )
    aligned_patch_points = np.asarray(
        npz[f"view{view_key}_patch_world_xyz"], dtype=np.float64,
    )
    raw_heat = np.asarray(
        npz[f"view{view_key}_heat_values"], dtype=np.float64,
    )

    if len(raw_heat) == len(raw_patch_points):
        patch_points = raw_patch_points
        patch_heat = raw_heat
    elif len(raw_heat) == len(aligned_patch_points):
        patch_points = raw_patch_points
        patch_heat = transfer_sparse_point_values_nn(
            raw_patch_points,
            aligned_patch_points,
            np.clip(raw_heat, 0.0, 1.0),
        )
    else:
        patch_points = aligned_patch_points
        patch_heat = raw_heat

    # Keep the comparison object cloud in the same frame as the patch points.
    # For these per-view asymmetric exports the raw patch points correspond to
    # the observed object surface from that view.
    object_points = raw_patch_points
    patch_pixels = np.full((len(patch_points), 2), -1, dtype=np.int32)

    pose = np.asarray(meta[f"pose{view_key}_cam_to_world"], dtype=np.float64)
    image_ref = meta[f"view{view_key}_image"]
    camera_origin = pose[:3, 3]
    scene_points = transform_points(load_depth_points_camera(image_ref), pose)
    if len(scene_points) == 0:
        scene_points = object_points
    object_name = meta.get(f"view{view_key}_object_name", source["prefix"])
    frame_id = source["frame_id"]
    object_tag = _safe_tag(str(object_name).lower())
    if view_key == 1:
        heatmap_path = SAM_OUTPUTS / f"rgb_{frame_id}_{object_tag}_pred_final_gray.png"
    else:
        heatmap_path = SAM_OUTPUTS / "asymmetric_view2_heatmaps" / f"rgb_{frame_id}_{object_tag}_pred_final_gray.png"
    if not heatmap_path.exists():
        heatmap_path = None

    return object_points, scene_points, patch_points, patch_pixels, patch_heat, camera_origin, heatmap_path


def update_grasp_config(prefix: str, frame_dir: Path) -> None:
    """
    Point grasp_config.py to the current adapted object folder.

    Three config lines are updated on every call:
      FRAME_DIR     — where the ply/npz files live
      FRAME_PREFIX  — filename stem (e.g. "blue_cup")
      ACTIVE_OBJECT — forced to None so grasp selection stays generic and
                      does not use object-specific tuning
    """
    config_path = CODE_ROOT / "grasp_config.py"
    text = config_path.read_text()
    new = text
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("FRAME_DIR ="):
            new = new.replace(line, f"FRAME_DIR = Path({str(frame_dir)!r})")
        if stripped.startswith("FRAME_PREFIX ="):
            new = new.replace(line, f'FRAME_PREFIX = "{prefix}"')
        if stripped.startswith("HEATMAP_DIR ="):
            new = new.replace(
                line,
                f"HEATMAP_DIR = Path({str(HEATMAP_DIR)!r})",
            )
        if stripped.startswith("ACTIVE_OBJECT") and "=" in stripped and "str | None" in stripped:
            new = new.replace(line, 'ACTIVE_OBJECT: str | None = None')
    config_path.write_text(new)


def write_adapted_object(
    out_dir: Path,
    prefix: str,
    frame_name: str,
    object_points: np.ndarray,
    scene_points: np.ndarray,
    patch_points: np.ndarray,
    patch_pixels: np.ndarray,
    patch_heat: np.ndarray,
    camera_origin: np.ndarray,
    heatmap_path: Path | None,
    object_flexible: bool = False,
    baseline_object_points: np.ndarray | None = None,
) -> Path:
    """Write one adapted object in the local grasp_work layout."""
    out_dir.mkdir(parents=True, exist_ok=True)

    object_normals = estimate_normals(object_points, k=min(20, max(2, len(object_points) - 1)))
    patch_normals = estimate_normals(patch_points, k=min(20, max(2, len(patch_points) - 1)))

    write_binary_ply(out_dir / f"{prefix}_object_cloud.ply", object_points, object_normals)
    if baseline_object_points is not None and len(baseline_object_points) > 0:
        baseline_normals = estimate_normals(
            baseline_object_points,
            k=min(20, max(2, len(baseline_object_points) - 1)),
        )
        write_binary_ply(
            out_dir / f"{prefix}_object_cloud_no_heatmap_dense.ply",
            baseline_object_points,
            baseline_normals,
        )
    scene_normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(scene_points), 1))
    write_binary_ply(out_dir / f"{prefix}_scene_cloud.ply", scene_points, scene_normals)
    write_binary_ply(out_dir / f"{prefix}_grasp_patch_cloud.ply", patch_points, patch_normals)

    np.savez(
        out_dir / f"{prefix}_grasp_patch_data.npz",
        frame_name=frame_name,
        camera_origin=np.asarray(camera_origin, dtype=np.float64),
        points=np.asarray(patch_points, dtype=np.float64),
        pixels=np.asarray(patch_pixels, dtype=np.int32),
        point_heat_values=np.asarray(np.clip(patch_heat, 0.0, 1.0), dtype=np.float64),
        object_flexible=np.asarray(bool(object_flexible), dtype=np.bool_),
    )

    metadata = {
        "frame_name": frame_name,
        "frame_prefix": prefix,
        "object_flexible": bool(object_flexible),
        "source_heatmap_path": str(heatmap_path) if heatmap_path is not None else None,
    }
    (out_dir / f"{prefix}_grasp_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    if heatmap_path is not None and heatmap_path.exists():
        HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(heatmap_path, HEATMAP_DIR / f"{frame_name}_pred_gray.png")

    return out_dir


def adapt_source(source: dict) -> tuple[str, Path, str]:
    """Load one upstream source, convert it, and write the adapted files."""
    prefix = source["prefix"]
    kind = source["kind"]
    source_label = source["source_label"]
    object_flexible = bool(source.get("object_flexible", flexible_for_source_name(prefix)))

    if kind == "root_heatmap_npz":
        data = load_root_heatmap_npz(source)
    elif kind in {"root_heatmap_pointcloud", "view2_heatmap_pointcloud"}:
        data = load_heatmap_pointcloud(source)
    elif kind == "symmetric_semantic":
        data = load_symmetric_semantic(source)
    elif kind == "asymmetric_fused":
        data = load_asymmetric_fused(source)
    elif kind == "asymmetric_view_patch":
        data = load_asymmetric_view_patch(source)
    elif kind == "root_single_view":
        data = load_root_single_view(source)
    else:
        raise ValueError(f"Unknown source kind: {kind}")

    object_points, scene_points, patch_points, patch_pixels, patch_heat, camera_origin, heatmap_path = data
    baseline_object_points = None
    if kind == "symmetric_semantic":
        obj_dir = Path(source["path"])
        metadata = json.loads((obj_dir / "metadata.json").read_text())
        if metadata.get("roll_model") and not metadata.get("semantic_cylinder"):
            baseline_object_points = build_roll_symmetric_object_cloud(obj_dir)
        elif metadata.get("semantic_cylinder"):
            baseline_object_points = build_semantic_cylinder_object_cloud(obj_dir, metadata)
    out_dir = GRASP_OBJ_DIR / prefix / _safe_tag(source.get("run_tag", source_label))
    frame_name = f"{prefix}_{_safe_tag(source.get('run_tag', source_label))}"
    out_dir = write_adapted_object(
        out_dir, prefix, frame_name, object_points, scene_points, patch_points, patch_pixels, patch_heat,
        camera_origin, heatmap_path, object_flexible=object_flexible, baseline_object_points=baseline_object_points,
    )

    print(
        f"  [adapt] {prefix:<18} source={source_label:<18} "
        f"flexible={str(object_flexible):<5} "
        f"object_pts={len(object_points):>5} patch_pts={len(patch_points):>5}"
    )
    return prefix, out_dir, source_label


def run_grasp_for(prefix: str, out_dir: Path) -> dict | None:
    """Run the scoring pipeline for one adapted object and save its report."""
    update_grasp_config(prefix, out_dir)

    res = subprocess.run(
        [sys.executable, str(CODE_ROOT / "run_grasps.py")],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"  [error] run_grasps.py failed for {prefix}")
        if res.stderr.strip():
            print(res.stderr[-600:])
        return None

    stdout_lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    for ln in stdout_lines[-4:]:
        print(f"   {ln}")

    scores = json.loads((PROJECT_ROOT / "grasp_scores.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grasp_scores.json").write_text(json.dumps(scores, indent=2))
    return scores


def run_visualize(prefix: str, out_dir: Path, run_tag: str, alias: str | None = None) -> None:
    """
    Save the best-grasp visualisation for the current object.

    Respect the user's chosen matplotlib backend:
      - MPLBACKEND=Agg   -> save PNG only
      - interactive      -> show window, then continue
    """
    import os

    res = subprocess.run(
        [sys.executable, str(CODE_ROOT / "visualize_grasp.py")],
        env=os.environ.copy(),
    )
    if res.returncode != 0:
        print(f"  [warn] visualize_grasp.py exit {res.returncode} for {prefix}")
        return
    src_png = PROJECT_ROOT / f"grasp_visualization_{prefix}.png"
    if src_png.exists():
        shutil.copyfile(src_png, out_dir / f"grasp_visualization_{prefix}_{_safe_tag(run_tag)}.png")
        if alias:
            shutil.copyfile(src_png, out_dir.parent / f"grasp_visualization_{prefix}_{_safe_tag(alias)}.png")


def _contact_set_distance_cm(a: dict | None, b: dict | None) -> float | None:
    if a is None or b is None:
        return None
    pa = np.array([c["point_xyz_m"] for c in a["contacts"]], dtype=np.float64)
    pb = np.array([c["point_xyz_m"] for c in b["contacts"]], dtype=np.float64)
    if len(pa) == 0 or len(pb) == 0:
        return None
    d_ab = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    chamfer = 0.5 * (d_ab.min(axis=1).mean() + d_ab.min(axis=0).mean())
    return round(float(chamfer) * 100.0, 3)


def _grasp_center_distance_cm(a: dict | None, b: dict | None) -> float | None:
    if a is None or b is None:
        return None
    pa = np.array([c["point_xyz_m"] for c in a["contacts"]], dtype=np.float64).mean(axis=0)
    pb = np.array([c["point_xyz_m"] for c in b["contacts"]], dtype=np.float64).mean(axis=0)
    return round(float(np.linalg.norm(pa - pb)) * 100.0, 3)


def _range_match_score(value_m: float, low_m: float, high_m: float) -> float:
    if low_m <= value_m <= high_m:
        return 1.0
    range_width = max(high_m - low_m, 1e-6)
    soft_margin = max(0.005, 0.25 * range_width, 0.15 * high_m)
    if value_m < low_m:
        return float(np.clip(1.0 - (low_m - value_m) / soft_margin, 0.0, 1.0))
    return float(np.clip(1.0 - (value_m - high_m) / soft_margin, 0.0, 1.0))


def build_posthoc_dimension_report(prefix: str, best_candidate: dict | None) -> dict | None:
    prior = get_object_dimensions(prefix)
    if best_candidate is None or prior is None:
        return None
    local_diameter_m = float(best_candidate["scores"]["local_diameter_mm"]) / 1000.0
    range_scores = [
        {
            "range_cm": [round(low * 100.0, 3), round(high * 100.0, 3)],
            "match_score": round(_range_match_score(local_diameter_m, low, high), 4),
        }
        for (low, high) in prior.grasp_diameter_ranges_m
    ]
    best_match = max((r["match_score"] for r in range_scores), default=0.0)
    return {
        "reference_object": prior.as_dict(),
        "selected_local_diameter_cm": round(local_diameter_m * 100.0, 3),
        "range_matches": range_scores,
        "best_match_score": round(best_match, 4),
    }


def main():
    source_index = collect_available_sources()
    evaluations = build_object_evaluations(source_index)
    if not evaluations:
        raise RuntimeError(f"No usable SAM3 outputs found under {SAM_OUTPUTS}")

    print(f"\nUsing SAM outputs from: {SAM_OUTPUTS}")
    print(f"Grasp evaluation mode: {GRASP_EVAL_MODE}")
    print(f"\n=== Grasp evaluation plan: {len(evaluations)} object(s) ===")
    for ev in evaluations:
        print(f"   {ev['prefix']:<18} class={ev['object_class']:<10} primary={ev['primary']['source_label']}")
    print()

    summary = []
    consistency_report: list[dict] = []
    for ev in evaluations:
        prefix = ev["prefix"]
        primary_source = ev["primary"]
        print(f"--- {prefix} ({ev['object_class']}) ---")

        _, primary_dir, primary_label = adapt_source(primary_source)
        primary_report = run_grasp_for(prefix, primary_dir)
        if primary_report is None:
            summary.append({"obj": prefix, "source": primary_label, "ok": False})
            print()
            continue
        primary_best = primary_report.get("best_candidate")
        if primary_best is None:
            summary.append({"obj": prefix, "source": primary_label, "ok": False})
            print("  [warn] no primary best_candidate was produced")
            print()
            continue
        primary_scores = primary_best["scores"]
        if primary_scores["valid"]:
            primary_alias = None if GRASP_EVAL_MODE == "primary_only" else "varianta_1"
            run_visualize(prefix, primary_dir, primary_source["run_tag"], alias=primary_alias)
        else:
            print(f"  [skip] primary visualization skipped for invalid grasp: {primary_label}")
        summary.append({
            "obj": prefix,
            "source": primary_label,
            "object_flexible": bool(primary_source.get("object_flexible", False)),
            "ok": True,
            "score": primary_scores["final_score"],
            "valid": primary_scores["valid"],
            "mode": primary_best["gripper_mode"],
            "rotation_deg": primary_best.get("rotation_deg", 0.0),
            "grip_type": primary_scores["grip_type"],
            "diameter_cm": primary_scores.get("local_diameter_cm", primary_scores["local_diameter_mm"] / 10.0),
        })

        comparison_items = []
        if GRASP_EVAL_MODE == "compare_views":
            variant2_saved = False
            for cmp_source in ev["comparisons"]:
                _, cmp_dir, cmp_label = adapt_source(cmp_source)
                cmp_report = run_grasp_for(prefix, cmp_dir)
                if cmp_report is None:
                    comparison_items.append({
                        "source": cmp_label,
                        "ok": False,
                    })
                    continue
                cmp_best = cmp_report.get("best_candidate")
                if cmp_best is None:
                    comparison_items.append({
                        "source": cmp_label,
                        "ok": False,
                        "reason": "no grasp candidate produced",
                    })
                    continue
                cmp_scores = cmp_best["scores"]
                if cmp_scores["valid"]:
                    if not variant2_saved:
                        run_visualize(
                            prefix,
                            cmp_dir,
                            cmp_source["run_tag"],
                            alias="varianta_2",
                        )
                        variant2_saved = True
                else:
                    print(f"  [skip] comparison visualization skipped for invalid grasp: {cmp_label}")
                comparison_items.append({
                    "source": cmp_label,
                    "ok": True,
                    "valid": cmp_scores["valid"],
                    "mode": cmp_best["gripper_mode"],
                    "grip_type": cmp_scores["grip_type"],
                    "diameter_cm": cmp_scores.get("local_diameter_cm", cmp_scores["local_diameter_mm"] / 10.0),
                    "score": cmp_scores["final_score"],
                    "contact_set_distance_cm": _contact_set_distance_cm(primary_best, cmp_best),
                    "grasp_center_distance_cm": _grasp_center_distance_cm(primary_best, cmp_best),
                })
                if variant2_saved:
                    break

        consistency_report.append({
            "object": prefix,
            "object_class": ev["object_class"],
            "object_flexible": bool(primary_source.get("object_flexible", False)),
            "primary": {
                "source": primary_label,
                "mode": primary_best["gripper_mode"],
                "grip_type": primary_scores["grip_type"],
                "valid": primary_scores["valid"],
                "score": primary_scores["final_score"],
                "diameter_cm": primary_scores.get("local_diameter_cm", primary_scores["local_diameter_mm"] / 10.0),
            },
            "comparisons": comparison_items,
            "posthoc_dimension_validation": build_posthoc_dimension_report(prefix, primary_best),
        })
        print()

    print("\n=== SUMMARY ===")
    print(f"{'object':<18}{'source':<20}{'mode':>10}{'rot':>8}{'grip':>14}{'Ø(cm)':>9}{'score':>9}{'valid':>8}")
    for item in summary:
        if not item["ok"]:
            print(f"{item['obj']:<18}{item['source']:<20} failed")
            continue
        print(
            f"{item['obj']:<18}{item['source']:<20}"
            f"{item['mode']:>10}{item['rotation_deg']:>8.0f}{item['grip_type']:>14}"
            f"{item['diameter_cm']:>9.2f}{item['score']:>9.4f}{str(item['valid']):>8}"
        )

    CONSISTENCY_REPORT_PATH.write_text(json.dumps(consistency_report, indent=2))
    print(f"\nConsistency report saved to: {CONSISTENCY_REPORT_PATH.name}")
    print("\nVisualizations saved as:")
    if GRASP_EVAL_MODE == "compare_views":
        print("  grasp_visualization_<object>_varianta_1.png")
        print("  grasp_visualization_<object>_varianta_2.png")
    else:
        print("  grasp_ready_objects/<object>/<source>/grasp_visualization_<object>_<source>.png")


if __name__ == "__main__":
    main()
