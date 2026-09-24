from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Literal
import json

import cv2
import numpy as np


DepthUnit = Literal["m", "mm"]
SymmetryAxisMode = Literal["vertical", "pca"]


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None = None
    height: int | None = None

    @staticmethod
    def from_json(path: Path) -> "CameraIntrinsics":
        data = json.loads(path.read_text())

        if "camera_matrix" in data:
            k = np.asarray(data["camera_matrix"], dtype=np.float64)
            return CameraIntrinsics(
                fx=float(k[0, 0]),
                fy=float(k[1, 1]),
                cx=float(k[0, 2]),
                cy=float(k[1, 2]),
                width=data.get("width"),
                height=data.get("height"),
            )

        return CameraIntrinsics(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            width=data.get("width"),
            height=data.get("height"),
        )

    def scaled(self, original_size: tuple[int, int], target_size: tuple[int, int]) -> "CameraIntrinsics":
        oh, ow = original_size
        th, tw = target_size
        sx = tw / float(ow)
        sy = th / float(oh)

        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=tw,
            height=th,
        )


@dataclass
class ReconstructionConfig:
    heatmap_threshold: float = 0.4
    min_component_area_px: int = 20
    max_depth_m: float = 3.0
    depth_unit: DepthUnit = "m"
    symmetry_axis_mode: SymmetryAxisMode = "vertical"
    export_npz: bool = True
    export_ply: bool = True


@dataclass
class PatchPointCloud:
    patch_id: int
    kind: Literal["original", "symmetric"]
    points_xyz: np.ndarray
    pixels_uv: np.ndarray
    heat_values: np.ndarray
    centroid_xyz: np.ndarray
    camera_origin_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def to_metadata(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "kind": self.kind,
            "num_points": int(len(self.points_xyz)),
            "centroid_xyz": self.centroid_xyz.astype(float).tolist(),
            "camera_origin_xyz": self.camera_origin_xyz.astype(float).tolist(),
        }


@dataclass
class ObjectReconstruction3D:
    image_path: str
    object_name: str
    flexible: bool
    camera_intrinsics: CameraIntrinsics
    camera_origin_xyz: np.ndarray
    object_pointcloud_xyz: np.ndarray
    object_pointcloud_rgb: np.ndarray | None
    heatmap_pointcloud_xyz: np.ndarray
    heatmap_values: np.ndarray
    original_patches: list[PatchPointCloud]
    symmetric_patches: list[PatchPointCloud]
    symmetry_axis_2d: dict[str, Any]

    @property
    def all_patches(self) -> list[PatchPointCloud]:
        return [*self.original_patches, *self.symmetric_patches]

    def metadata(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "object_name": self.object_name,
            "flexible": bool(self.flexible),
            "object_flexible": bool(self.flexible),
            "camera_intrinsics": asdict(self.camera_intrinsics),
            "camera_origin_xyz": self.camera_origin_xyz.astype(float).tolist(),
            "num_object_points": int(len(self.object_pointcloud_xyz)),
            "num_heatmap_points": int(len(self.heatmap_pointcloud_xyz)),
            "symmetry_axis_2d": self.symmetry_axis_2d,
            "patches": [p.to_metadata() for p in self.all_patches],
        }

    def save(self, output_dir: Path, stem: str) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / f"{stem}_reconstruction_meta.json").write_text(
            json.dumps(self.metadata(), indent=2),
            encoding="utf-8",
        )

        object_rgb = (
            self.object_pointcloud_rgb
            if self.object_pointcloud_rgb is not None
            else np.full((len(self.object_pointcloud_xyz), 3), 180, dtype=np.uint8)
        )

        write_ply(output_dir / f"{stem}_object_cloud.ply", self.object_pointcloud_xyz, object_rgb)

        if len(self.heatmap_pointcloud_xyz):
            heat_rgb = heat_to_rgb_uint8(self.heatmap_values)

            write_ply(
                output_dir / f"{stem}_heatmap_cloud.ply",
                self.heatmap_pointcloud_xyz,
                heat_rgb,
            )

            np.savez_compressed(
                output_dir / f"{stem}_heatmap_cloud.npz",
                points_xyz=self.heatmap_pointcloud_xyz.astype(np.float32),
                heat_values=self.heatmap_values.astype(np.float32),
                camera_origin_xyz=self.camera_origin_xyz.astype(np.float32),
                object_flexible=np.asarray(bool(self.flexible), dtype=np.bool_),
            )

        for patch in self.all_patches:
            suffix = f"patch_{patch.patch_id:02d}_{patch.kind}"
            rgb = heat_to_rgb_uint8(patch.heat_values)

            write_ply(
                output_dir / f"{stem}_{suffix}.ply",
                patch.points_xyz,
                rgb,
            )

            np.savez_compressed(
                output_dir / f"{stem}_{suffix}.npz",
                points_xyz=patch.points_xyz.astype(np.float32),
                pixels_uv=patch.pixels_uv.astype(np.int32),
                heat_values=patch.heat_values.astype(np.float32),
                centroid_xyz=patch.centroid_xyz.astype(np.float32),
                camera_origin_xyz=patch.camera_origin_xyz.astype(np.float32),
                object_flexible=np.asarray(bool(self.flexible), dtype=np.bool_),
            )


def load_depth(path: Path, target_size: tuple[int, int], depth_unit: DepthUnit = "m") -> np.ndarray:
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

    th, tw = target_size
    if depth.shape[:2] != (th, tw):
        depth = cv2.resize(depth, (tw, th), interpolation=cv2.INTER_NEAREST)

    return depth.astype(np.float32)


def backproject_pixels(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    intr: CameraIntrinsics,
) -> np.ndarray:
    x = (u.astype(np.float32) - intr.cx) * z.astype(np.float32) / intr.fx
    y = (v.astype(np.float32) - intr.cy) * z.astype(np.float32) / intr.fy
    return np.stack([x, y, z.astype(np.float32)], axis=1)

def estimate_symmetry_plane_normal_from_object(points_xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32)

    if len(pts) < 3:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    center = pts.mean(axis=0)
    centered = pts - center[None, :]

    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    # Pentru obiectul văzut din cameră:
    # vt[0] = axa cea mai lungă
    # vt[1] = axa secundară
    # vt[2] = axa cea mai subțire / normală aproximativă
    #
    # Pentru oglindire stânga-dreapta pe obiect, de obicei normalul planului
    # e axa secundară în planul obiectului, nu X global.
    n = vt[1].astype(np.float32)

    n = n / max(float(np.linalg.norm(n)), 1e-6)

    return n

def reconstruct_symmetric_object(
    *,
    image_path: Path,
    object_name: str,
    rgb: np.ndarray,
    object_mask: np.ndarray,
    heatmap: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    config: ReconstructionConfig | None = None,
    object_flexible: bool = False,
) -> ObjectReconstruction3D:
    cfg = config or ReconstructionConfig()

    mask = object_mask.astype(bool)
    valid_depth = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= cfg.max_depth_m)

    obj_xyz, obj_rgb, _ = build_masked_cloud(
        rgb=rgb,
        mask=mask,
        heatmap=None,
        depth_m=depth_m,
        valid_depth=valid_depth,
        intrinsics=intrinsics,
    )

    object_center = (
        obj_xyz.mean(axis=0).astype(np.float32)
        if len(obj_xyz)
        else np.zeros(3, dtype=np.float32)
    )

    heatmap_xyz, _, heatmap_values = build_masked_cloud(
        rgb=rgb,
        mask=mask & np.isfinite(heatmap) & (heatmap > 0.0),
        heatmap=heatmap,
        depth_m=depth_m,
        valid_depth=valid_depth,
        intrinsics=intrinsics,
    )

    patch_mask = (heatmap >= cfg.heatmap_threshold) & mask & valid_depth
    components = connected_components(patch_mask, cfg.min_component_area_px)

    axis_2d = estimate_symmetry_axis_2d(mask, mode=cfg.symmetry_axis_mode)

    original_patches: list[PatchPointCloud] = []
    symmetric_patches: list[PatchPointCloud] = []

    # Plan vertical de simetrie în 3D:
    # punct pe plan = centrul obiectului
    # normala planului = axa X a camerei
    # Formula: x_sym = 2 * center_x - x, y_sym = y, z_sym = z
    plane_point = object_center
    plane_normal = estimate_symmetry_plane_normal_from_object(obj_xyz)

    for patch_id, comp in enumerate(components, start=1):
        pv, pu = np.where(comp)

        if len(pu) == 0:
            continue

        pz = depth_m[pv, pu]
        pxyz = backproject_pixels(pu, pv, pz, intrinsics)
        heat_values = heatmap[pv, pu].astype(np.float32)

        original_patch = make_patch(
            patch_id=patch_id,
            kind="original",
            xyz=pxyz,
            u=pu,
            v=pv,
            heat_values=heat_values,
        )
        original_patches.append(original_patch)

        symmetric_xyz = reflect_points_across_plane(
            points_xyz=pxyz,
            plane_point=plane_point,
            plane_normal=plane_normal,
        )

        symmetric_patch = make_patch(
            patch_id=patch_id,
            kind="symmetric",
            xyz=symmetric_xyz,
            u=np.zeros(len(symmetric_xyz), dtype=np.int32),
            v=np.zeros(len(symmetric_xyz), dtype=np.int32),
            heat_values=heat_values.copy(),
        )
        symmetric_patches.append(symmetric_patch)

    axis_2d["symmetry_3d"] = {
        "type": "vertical_x_plane_reflection",
        "object_center_xyz": object_center.astype(float).tolist(),
        "plane_point_xyz": plane_point.astype(float).tolist(),
        "plane_normal_xyz": plane_normal.astype(float).tolist(),
        "formula": "x_sym = 2 * center_x - x, y_sym = y, z_sym = z",
        "note": (
            "Symmetric patch is generated point-by-point by reflecting the original 3D patch "
            "across a vertical symmetry plane passing through the object center."
        ),
    }

    return ObjectReconstruction3D(
        image_path=str(image_path),
        object_name=object_name,
        flexible=bool(object_flexible),
        camera_intrinsics=intrinsics,
        camera_origin_xyz=np.zeros(3, dtype=np.float32),
        object_pointcloud_xyz=obj_xyz.astype(np.float32),
        object_pointcloud_rgb=obj_rgb,
        heatmap_pointcloud_xyz=heatmap_xyz.astype(np.float32),
        heatmap_values=heatmap_values.astype(np.float32),
        original_patches=original_patches,
        symmetric_patches=symmetric_patches,
        symmetry_axis_2d=axis_2d,
    )


def reconstruct_asymmetric_object(
    *,
    image_path_1: Path,
    image_path_2: Path,
    object_name: str,
    rgb1: np.ndarray,
    rgb2: np.ndarray,
    object_mask1: np.ndarray,
    object_mask2: np.ndarray,
    heatmap1: np.ndarray,
    heatmap2: np.ndarray,
    depth_m1: np.ndarray,
    depth_m2: np.ndarray,
    intrinsics: CameraIntrinsics,
    pose1: np.ndarray,
    pose2: np.ndarray,
    config: ReconstructionConfig | None = None,
    object_flexible: bool = False,
) -> ObjectReconstruction3D:
    cfg = config or ReconstructionConfig()

    mask1 = object_mask1.astype(bool)
    mask2 = object_mask2.astype(bool)

    valid_depth1 = np.isfinite(depth_m1) & (depth_m1 > 0.0) & (depth_m1 <= cfg.max_depth_m)
    valid_depth2 = np.isfinite(depth_m2) & (depth_m2 > 0.0) & (depth_m2 <= cfg.max_depth_m)

    obj1_xyz, obj1_rgb, _ = build_masked_cloud(
        rgb=rgb1,
        mask=mask1,
        heatmap=None,
        depth_m=depth_m1,
        valid_depth=valid_depth1,
        intrinsics=intrinsics,
    )

    obj2_xyz, obj2_rgb, _ = build_masked_cloud(
        rgb=rgb2,
        mask=mask2,
        heatmap=None,
        depth_m=depth_m2,
        valid_depth=valid_depth2,
        intrinsics=intrinsics,
    )

    transform_2_to_1 = relative_transform_source_to_target(
        source_pose=pose2,
        target_pose=pose1,
    )

    obj2_in_1 = transform_points(obj2_xyz, transform_2_to_1)
    object_xyz = np.vstack([obj1_xyz, obj2_in_1]).astype(np.float32)

    if obj1_rgb is not None and obj2_rgb is not None:
        object_rgb = np.vstack([obj1_rgb, obj2_rgb]).astype(np.uint8)
    else:
        object_rgb = None

    heat1_xyz, _, heat1_values = build_masked_cloud(
        rgb=rgb1,
        mask=mask1 & np.isfinite(heatmap1) & (heatmap1 > 0.0),
        heatmap=heatmap1,
        depth_m=depth_m1,
        valid_depth=valid_depth1,
        intrinsics=intrinsics,
    )

    heat2_xyz, _, heat2_values = build_masked_cloud(
        rgb=rgb2,
        mask=mask2 & np.isfinite(heatmap2) & (heatmap2 > 0.0),
        heatmap=heatmap2,
        depth_m=depth_m2,
        valid_depth=valid_depth2,
        intrinsics=intrinsics,
    )

    heat2_in_1 = transform_points(heat2_xyz, transform_2_to_1)
    heatmap_xyz = np.vstack([heat1_xyz, heat2_in_1]).astype(np.float32)
    heatmap_values = np.concatenate([heat1_values, heat2_values]).astype(np.float32)

    patch1 = make_heatmap_patch(
        patch_id=1,
        kind="original",
        mask=mask1,
        heatmap=heatmap1,
        depth_m=depth_m1,
        valid_depth=valid_depth1,
        intrinsics=intrinsics,
        cfg=cfg,
        transform=None,
    )

    patch2 = make_heatmap_patch(
        patch_id=2,
        kind="original",
        mask=mask2,
        heatmap=heatmap2,
        depth_m=depth_m2,
        valid_depth=valid_depth2,
        intrinsics=intrinsics,
        cfg=cfg,
        transform=transform_2_to_1,
    )

    original_patches = [p for p in [patch1, patch2] if p is not None]

    camera_2_origin_in_camera_1 = transform_points(
        np.zeros((1, 3), dtype=np.float32),
        transform_2_to_1,
    )[0]

    metadata = {
        "mode": "asymmetric_two_view_fusion",
        "image_path_1": str(image_path_1),
        "image_path_2": str(image_path_2),
        "camera_1_origin_xyz": [0.0, 0.0, 0.0],
        "camera_2_origin_in_camera_1_xyz": camera_2_origin_in_camera_1.astype(float).tolist(),
        "pose1": pose1.astype(float).tolist(),
        "pose2": pose2.astype(float).tolist(),
        "transform_camera2_to_camera1": transform_2_to_1.astype(float).tolist(),
        "note": "Frame 2 pointcloud was transformed into frame 1 and fused with frame 1.",
    }

    return ObjectReconstruction3D(
        image_path=f"{image_path_1} + {image_path_2}",
        object_name=object_name,
        flexible=bool(object_flexible),
        camera_intrinsics=intrinsics,
        camera_origin_xyz=np.zeros(3, dtype=np.float32),
        object_pointcloud_xyz=object_xyz.astype(np.float32),
        object_pointcloud_rgb=object_rgb,
        heatmap_pointcloud_xyz=heatmap_xyz.astype(np.float32),
        heatmap_values=heatmap_values.astype(np.float32),
        original_patches=original_patches,
        symmetric_patches=[],
        symmetry_axis_2d=metadata,
    )


def build_masked_cloud(
    *,
    rgb: np.ndarray | None,
    mask: np.ndarray,
    heatmap: np.ndarray | None,
    depth_m: np.ndarray,
    valid_depth: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    pixels = mask.astype(bool) & valid_depth
    v, u = np.where(pixels)

    if len(u) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            None,
            np.zeros((0,), dtype=np.float32),
        )

    xyz = backproject_pixels(u, v, depth_m[v, u], intrinsics).astype(np.float32)

    rgb_values = None
    if rgb is not None:
        rgb_values = rgb[v, u].astype(np.uint8)

    heat_values = (
        heatmap[v, u].astype(np.float32)
        if heatmap is not None
        else np.zeros((len(u),), dtype=np.float32)
    )

    return xyz, rgb_values, heat_values


def make_heatmap_patch(
    *,
    patch_id: int,
    kind: Literal["original", "symmetric"],
    mask: np.ndarray,
    heatmap: np.ndarray,
    depth_m: np.ndarray,
    valid_depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    cfg: ReconstructionConfig,
    transform: np.ndarray | None = None,
) -> PatchPointCloud | None:
    patch_mask = (heatmap >= cfg.heatmap_threshold) & mask.astype(bool) & valid_depth
    components = connected_components(patch_mask, cfg.min_component_area_px)

    if not components:
        return None

    comp = max(components, key=lambda c: int(np.sum(c)))
    v, u = np.where(comp)

    if len(u) == 0:
        return None

    xyz = backproject_pixels(u, v, depth_m[v, u], intrinsics).astype(np.float32)

    if transform is not None:
        xyz = transform_points(xyz, transform).astype(np.float32)

    return make_patch(
        patch_id=patch_id,
        kind=kind,
        xyz=xyz,
        u=u,
        v=v,
        heat_values=heatmap[v, u].astype(np.float32),
    )


def reflect_points_across_plane(
    *,
    points_xyz: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32)
    p0 = np.asarray(plane_point, dtype=np.float32)
    n = np.asarray(plane_normal, dtype=np.float32)

    n = n / max(float(np.linalg.norm(n)), 1e-6)

    rel = pts - p0[None, :]
    signed_distance = rel @ n
    reflected = pts - 2.0 * signed_distance[:, None] * n[None, :]

    return reflected.astype(np.float32)


def transform_points(points_xyz: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)

    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    ones = np.ones((len(points), 1), dtype=np.float32)
    homogeneous = np.concatenate([points, ones], axis=1)
    transformed = (np.asarray(transform_4x4, dtype=np.float32) @ homogeneous.T).T

    return transformed[:, :3].astype(np.float32)


def relative_transform_source_to_target(
    *,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
) -> np.ndarray:
    source_pose = np.asarray(source_pose, dtype=np.float32)
    target_pose = np.asarray(target_pose, dtype=np.float32)

    return (np.linalg.inv(target_pose) @ source_pose).astype(np.float32)


def connected_components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )

    comps: list[np.ndarray] = []

    for label in range(1, num_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            comps.append(labels == label)

    return comps


def estimate_symmetry_axis_2d(mask: np.ndarray, mode: SymmetryAxisMode = "vertical") -> dict[str, Any]:
    ys, xs = np.where(mask.astype(bool))

    if len(xs) == 0:
        raise ValueError("Cannot estimate symmetry axis from an empty mask.")

    cx = float(xs.mean())
    cy = float(ys.mean())

    if mode == "vertical":
        direction = np.array([0.0, 1.0], dtype=np.float32)
    else:
        coords = np.stack(
            [
                xs.astype(np.float32) - cx,
                ys.astype(np.float32) - cy,
            ],
            axis=1,
        )

        _, _, vt = np.linalg.svd(coords, full_matrices=False)
        direction = vt[0].astype(np.float32)

        if direction[1] < 0:
            direction *= -1.0

    direction = direction / max(float(np.linalg.norm(direction)), 1e-6)

    return {
        "mode": mode,
        "point_uv": [cx, cy],
        "direction_uv": [float(direction[0]), float(direction[1])],
    }


def make_patch(
    patch_id: int,
    kind: Literal["original", "symmetric"],
    xyz: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    heat_values: np.ndarray,
) -> PatchPointCloud:
    xyz = np.asarray(xyz, dtype=np.float32)
    u = np.asarray(u, dtype=np.int32)
    v = np.asarray(v, dtype=np.int32)
    heat_values = np.asarray(heat_values, dtype=np.float32)

    centroid = (
        xyz.mean(axis=0).astype(np.float32)
        if len(xyz)
        else np.zeros(3, dtype=np.float32)
    )

    uv = np.stack([u, v], axis=1) if len(u) else np.zeros((0, 2), dtype=np.int32)

    return PatchPointCloud(
        patch_id=patch_id,
        kind=kind,
        points_xyz=xyz.astype(np.float32),
        pixels_uv=uv.astype(np.int32),
        heat_values=heat_values.astype(np.float32),
        centroid_xyz=centroid,
    )


def heat_to_rgb_uint8(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)

    if values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    norm = (values - values.min()) / max(float(values.max() - values.min()), 1e-6)

    red = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    green = np.clip((1.0 - np.abs(norm - 0.5) * 2.0) * 255.0, 0, 255).astype(np.uint8)
    blue = np.clip((1.0 - norm) * 255.0, 0, 255).astype(np.uint8)

    return np.stack([red, green, blue], axis=1)


def write_ply(
    path: Path,
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray | None = None,
) -> None:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)

    if colors_rgb is None:
        colors_rgb = np.full((len(points_xyz), 3), 180, dtype=np.uint8)

    colors_rgb = np.asarray(colors_rgb, dtype=np.uint8)

    if len(colors_rgb) != len(points_xyz):
        raise ValueError("colors_rgb must have the same length as points_xyz")

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points_xyz, colors_rgb):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )
