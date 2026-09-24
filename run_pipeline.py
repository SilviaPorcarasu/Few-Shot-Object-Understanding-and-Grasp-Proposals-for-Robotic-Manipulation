from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.pipeline import PipelineConfig, SymmetryAwareSamPipeline
from src.object_catalog import symmetry_for_name

from run_symmetric_patch_reconstruction import (
    prepare_depth_and_intrinsics,
    reconstruct_one_object,
    safe_name,
)

from run_asymmetric_patch_reconstruction import (
    prepare_depth_and_intrinsics as prepare_asym_depth_and_intrinsics,
    load_pose_matrix,
    collect_asymmetric_view_results,
    save_fused_object,
    visualize_fused_object as visualize_asymmetric_object_dir,
)

def save_heatmap_pointclouds(
    *,
    results,
    depth_m: np.ndarray,
    intr,
    output_dir: Path,
    max_depth_m: float,
) -> None:
    out_dir = output_dir / "heatmap_pointclouds"
    out_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        mask = result.sam_mask.astype(bool)
        heatmap = np.asarray(result.final_heatmap, dtype=np.float32)

        valid = (
            mask
            & np.isfinite(depth_m)
            & (depth_m > 0.0)
            & (depth_m <= max_depth_m)
        )

        v, u = np.where(valid)
        print(f"[HEATMAP POINTCLOUD] {result.object_name}: valid points = {len(u)}")
        if len(u) == 0:
            print(f"[WARNING] No valid heatmap pointcloud for {result.object_name!r}")
            continue

        z = depth_m[v, u].astype(np.float32)

        x = (u.astype(np.float32) - intr.cx) * z / intr.fx
        y = (v.astype(np.float32) - intr.cy) * z / intr.fy

        points = np.stack([x, y, z], axis=1).astype(np.float32)

        heat_values = heatmap[v, u]
        heat_norm = heat_values.copy()

        if float(heat_norm.max() - heat_norm.min()) > 1e-6:
            heat_norm = (heat_norm - heat_norm.min()) / (heat_norm.max() - heat_norm.min())
        else:
            heat_norm = np.ones_like(heat_norm)

        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[:, 0] = np.clip(255 * heat_norm, 0, 255).astype(np.uint8)
        colors[:, 1] = np.clip(80 * (1.0 - heat_norm), 0, 255).astype(np.uint8)
        colors[:, 2] = np.clip(255 * (1.0 - heat_norm), 0, 255).astype(np.uint8)

        obj_name = safe_name(result.object_name)
        stem = result.image_path.stem

        write_path = out_dir / f"{stem}_{obj_name}_heatmap_pointcloud.ply"

        with write_path.open("w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for p, c in zip(points, colors):
                f.write(
                    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
                )

        print(f"Saved heatmap pointcloud: {write_path}")
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=heat_norm,
            s=3,
            alpha=0.9,
        )

        center = points.mean(axis=0)
        max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2
        max_range = max(max_range, 0.01) * 1.15

        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{stem}_{obj_name}: heatmap pointcloud")

        ax.view_init(elev=-90, azim=-90)

        png_path = out_dir / f"{stem}_{obj_name}_heatmap_pointcloud.png"
        plt.tight_layout()
        plt.savefig(png_path, dpi=180)
        plt.close(fig)

        print(f"Saved heatmap visualization: {png_path}")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot detection -> symmetry routing -> SAM3 -> heatmap pipeline"
    )

    parser.add_argument("--heatmap-model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--second-input", type=Path, default=None)

    parser.add_argument("--output-dir", type=Path, default=Path("sam3_then_infer_outputs"))
    parser.add_argument("--save-outputs", action="store_true")

    parser.add_argument("--depth", type=Path, default=None)
    parser.add_argument("--second-depth", type=Path, default=None)
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument("--pose", type=Path, default=None)
    parser.add_argument("--second-pose", type=Path, default=None)
    parser.add_argument("--third-input", type=Path, default=None)
    parser.add_argument("--third-depth", type=Path, default=None)
    parser.add_argument("--third-pose", type=Path, default=None)
    parser.add_argument("--third-object-key", type=str, default="hand_cream_tube")

    parser.add_argument("--depth-unit", type=str, default="m", choices=["m", "mm"])
    parser.add_argument("--reconstruction-min-component-area", type=int, default=20)
    parser.add_argument("--reconstruction-max-depth", type=float, default=3.0)
    parser.add_argument("--symmetry-axis-mode", type=str, default="vertical", choices=["vertical", "pca"])
    parser.add_argument("--save-heatmap-pointclouds", action="store_true")
    parser.add_argument("--symmetric-input", type=Path, default=None)
    parser.add_argument("--symmetric-depth", type=Path, default=None)

    # semantic symmetric reconstruction
    parser.add_argument("--symmetric-3d-semantic", action="store_true")
    parser.add_argument("--patch-source", choices=["heatmap", "mask"], default="heatmap")
    parser.add_argument("--patch-dilate-kernel", type=int, default=0)
    parser.add_argument("--patch-dilate-iterations", type=int, default=0)
    parser.add_argument("--patch-keep-percentile", type=float, default=100.0)
    parser.add_argument("--center-depth-offset-ratio", type=float, default=0.5)
    parser.add_argument("--depth-window", type=float, default=0.08)

    # asymmetric two-view reconstruction
    parser.add_argument("--asymmetric-3d-twoview", action="store_true")
    parser.add_argument("--visualize-asymmetric-3d", action="store_true")
    parser.add_argument("--show-asymmetric-3d", action="store_true")
    parser.add_argument(
        "--pose-convention",
        choices=["camera_to_world", "world_to_camera"],
        default="world_to_camera",
    )
    # visualization
    parser.add_argument("--visualize-symmetric-3d", action="store_true")
    parser.add_argument("--show-symmetric-3d", action="store_true")

    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--detector-device", type=str, default=None)
    parser.add_argument("--sam-device", type=str, default=None)
    parser.add_argument("--heatmap-device", type=str, default=None)

    parser.add_argument("--detector-backend", type=str, default="grounding-dino", choices=["grounding-dino", "owlvit"])
    parser.add_argument("--detector-model-id", type=str, default=None)
    parser.add_argument("--detector-threshold", type=float, default=0.10)
    parser.add_argument("--detector-text-threshold", type=float, default=0.20)
    parser.add_argument("--detector-label-batch-size", type=int, default=20)
    parser.add_argument("--skip-asymmetric-first-image", action="store_true")
    parser.add_argument("--detector-top-k", type=int, default=None)

    parser.add_argument("--sam-model-id", type=str, default="facebook/sam3")
    parser.add_argument("--sam-score-threshold", type=float, default=0.5)
    parser.add_argument("--sam-mask-threshold", type=float, default=0.5)

    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--mask-background", type=str, default="black", choices=["black", "white"])

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
    bounded_args = [
        "threshold",
        "overlay_alpha",
        "sam_score_threshold",
        "sam_mask_threshold",
        "edge_shift_strength",
        "preserve_original",
    ]

    for name in bounded_args:
        value = getattr(args, name)
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")

    if not args.input.exists():
        raise RuntimeError(f"Input image not found: {args.input}")

    if args.second_input is not None and not args.second_input.exists():
        raise RuntimeError(f"Second input image not found: {args.second_input}")

    if args.third_input is not None and not args.third_input.exists():
        raise RuntimeError(f"Third input image not found: {args.third_input}")

    if (args.depth is None) ^ (args.intrinsics is None):
        raise RuntimeError("3D reconstruction needs both --depth and --intrinsics, or neither.")

    if args.depth is not None and not args.depth.exists():
        raise RuntimeError(f"Depth map not found: {args.depth}")

    if args.intrinsics is not None and not args.intrinsics.exists():
        raise RuntimeError(f"Intrinsics JSON not found: {args.intrinsics}")

    if args.symmetric_3d_semantic and (args.depth is None or args.intrinsics is None):
        raise RuntimeError("--symmetric-3d-semantic needs --depth and --intrinsics.")

    if args.visualize_symmetric_3d and not args.symmetric_3d_semantic:
        raise RuntimeError("--visualize-symmetric-3d needs --symmetric-3d-semantic.")

    if args.second_depth is not None and args.second_input is None:
        raise RuntimeError("--second-depth was provided, but --second-input is missing.")

    if args.second_depth is not None and not args.second_depth.exists():
        raise RuntimeError(f"Second depth map not found: {args.second_depth}")

    if args.third_depth is not None and args.third_input is None:
        raise RuntimeError("--third-depth was provided, but --third-input is missing.")

    if args.third_depth is not None and not args.third_depth.exists():
        raise RuntimeError(f"Third depth map not found: {args.third_depth}")

    if args.pose is not None and not args.pose.exists():
        raise RuntimeError(f"Pose file not found: {args.pose}")

    if args.second_pose is not None and not args.second_pose.exists():
        raise RuntimeError(f"Second pose file not found: {args.second_pose}")

    if args.third_pose is not None and args.third_input is None:
        raise RuntimeError("--third-pose was provided, but --third-input is missing.")

    if args.third_pose is not None and not args.third_pose.exists():
        raise RuntimeError(f"Third pose file not found: {args.third_pose}")

    if args.symmetric_input is not None and not args.symmetric_input.exists():
        raise RuntimeError(f"Symmetric input image not found: {args.symmetric_input}")

    if args.symmetric_depth is not None and not args.symmetric_depth.exists():
        raise RuntimeError(f"Symmetric depth map not found: {args.symmetric_depth}")

    if args.symmetric_3d_semantic:
        if args.symmetric_input is None:
            args.symmetric_input = args.input
        if args.symmetric_depth is None:
            args.symmetric_depth = args.depth
    asymmetric_pose_args = [args.second_input, args.second_depth, args.pose, args.second_pose]

    if args.asymmetric_3d_twoview:
        missing = []
        if args.second_input is None:
            missing.append("--second-input")
        if args.depth is None:
            missing.append("--depth")
        if args.second_depth is None:
            missing.append("--second-depth")
        if args.intrinsics is None:
            missing.append("--intrinsics")
        if args.pose is None:
            missing.append("--pose")
        if args.second_pose is None:
            missing.append("--second-pose")

        if missing:
            raise RuntimeError(
                "--asymmetric-3d-twoview needs: "
                "--input, --second-input, --depth, --second-depth, "
                "--intrinsics, --pose, --second-pose. Missing: "
                + ", ".join(missing)
            )

    if args.visualize_asymmetric_3d and not args.asymmetric_3d_twoview:
        raise RuntimeError("--visualize-asymmetric-3d needs --asymmetric-3d-twoview.")

    third_view_args = [args.third_input, args.third_depth, args.third_pose]
    if any(x is not None for x in third_view_args):
        missing = []
        if args.third_input is None:
            missing.append("--third-input")
        if args.third_depth is None:
            missing.append("--third-depth")
        if args.third_pose is None:
            missing.append("--third-pose")
        if args.intrinsics is None:
            missing.append("--intrinsics")

        if missing:
            raise RuntimeError(
                "Third-view object fusion needs all of: "
                "--third-input, --third-depth, --third-pose, --intrinsics. Missing: "
                + ", ".join(missing)
            )
    if any(x is not None for x in asymmetric_pose_args):
        missing = []
        if args.second_input is None:
            missing.append("--second-input")
        if args.depth is None:
            missing.append("--depth")
        if args.second_depth is None:
            missing.append("--second-depth")
        if args.intrinsics is None:
            missing.append("--intrinsics")
        if args.pose is None:
            missing.append("--pose")
        if args.second_pose is None:
            missing.append("--second-pose")

        if missing:
            raise RuntimeError(
                "Asymmetric two-view reconstruction needs all of: "
                "--input, --second-input, --depth, --second-depth, "
                "--intrinsics, --pose, --second-pose. Missing: "
                + ", ".join(missing)
            )


def visualize_symmetric_object_dir(
    object_dir: Path,
    patch_id: int = 1,
    show: bool = False,
) -> None:
    patch_file = object_dir / f"patch_{patch_id:02d}.npz"

    if not patch_file.exists():
        print(f"[WARNING] Missing patch file for visualization: {patch_file}")
        return

    data = np.load(patch_file, allow_pickle=True)

    orig = data["original_points_xyz"].astype(np.float32)
    sym = data["symmetric_points_xyz"].astype(np.float32)

    orig_centroid = data["centroid_original_xyz"].astype(np.float32)
    sym_centroid = data["centroid_symmetric_xyz"].astype(np.float32)

    orig = orig[np.isfinite(orig).all(axis=1)]
    sym = sym[np.isfinite(sym).all(axis=1)]

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        orig[:, 0],
        orig[:, 1],
        orig[:, 2],
        s=4,
        c="limegreen",
        alpha=1.0,
        label="original patch",
    )

    ax.scatter(
        sym[:, 0],
        sym[:, 1],
        sym[:, 2],
        s=12,
        c="cyan",
        alpha=0.9,
        label="symmetric patch",
    )

    ax.scatter(
        orig_centroid[0],
        orig_centroid[1],
        orig_centroid[2],
        s=250,
        c="red",
        marker="o",
        edgecolors="black",
        linewidths=1.5,
        label="original centroid",
    )

    ax.scatter(
        sym_centroid[0],
        sym_centroid[1],
        sym_centroid[2],
        s=320,
        c="magenta",
        marker="X",
        edgecolors="black",
        linewidths=1.5,
        label="symmetric centroid",
    )

    all_pts = np.vstack([
        orig,
        sym,
        orig_centroid.reshape(1, 3),
        sym_centroid.reshape(1, 3),
    ])

    center = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2

    if max_range < 1e-4:
        max_range = 0.01

    max_range *= 1.15

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(f"{object_dir.name}: original patch + symmetric patch")
    ax.legend(fontsize=10)

    # aproximativ perspectiva camerei
    ax.view_init(elev=25, azim=-60)

    plt.tight_layout()

    out_3d = object_dir / f"patch_{patch_id:02d}_visualization_3d.png"
    plt.savefig(out_3d, dpi=180)

    if show:
        plt.show()

    plt.close(fig)

    debug_img_path = object_dir / "debug_mask_patch_semantic_model.png"

    if debug_img_path.exists():
        debug_img = cv2.imread(str(debug_img_path))
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.imshow(debug_img)
        ax2.set_title("Mask + patch 2D overlay")
        ax2.axis("off")

        plt.tight_layout()

        out_2d = object_dir / "patch_overlay_2d.png"
        plt.savefig(out_2d, dpi=180)

        if show:
            plt.show()

        plt.close(fig2)

        print(f"Saved 2D visualization to: {out_2d}")

    print(f"Saved 3D visualization to: {out_3d}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    config = PipelineConfig(
        image_size=(args.height, args.width),
        device=args.device,
        detector_device=args.detector_device,
        sam_device=args.sam_device,
        heatmap_device=args.heatmap_device,
        detector_model_id=args.detector_model_id
        or (
            "IDEA-Research/grounding-dino-base"
            if args.detector_backend == "grounding-dino"
            else "google/owlvit-base-patch32"
        ),
        detector_backend=args.detector_backend,
        detector_threshold=args.detector_threshold,
        detector_text_threshold=args.detector_text_threshold,
        detector_label_batch_size=args.detector_label_batch_size,
        run_asymmetric_on_first_image=not args.skip_asymmetric_first_image,
        detector_top_k=args.detector_top_k,
        sam_model_id=args.sam_model_id,
        sam_score_threshold=args.sam_score_threshold,
        sam_mask_threshold=args.sam_mask_threshold,
        heatmap_threshold=args.threshold,
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
        depth_path=args.depth,
        second_depth_path=args.second_depth,
        intrinsics_path=args.intrinsics,
        pose_path=args.pose,
        second_pose_path=args.second_pose,
        depth_unit=args.depth_unit,
        reconstruction_min_component_area_px=args.reconstruction_min_component_area,
        reconstruction_max_depth_m=args.reconstruction_max_depth,
        symmetry_axis_mode=args.symmetry_axis_mode,
    )

    pipeline = SymmetryAwareSamPipeline(args.heatmap_model, config)
    predictions, results = pipeline.run(args.input, second_input_image=args.second_input)
    if args.save_heatmap_pointclouds:
        if args.depth is None or args.intrinsics is None:
            raise RuntimeError("--save-heatmap-pointclouds needs --depth and --intrinsics.")

        heat_depth_m, heat_intr = prepare_depth_and_intrinsics(
            args.depth,
            args.intrinsics,
            (args.height, args.width),
            args.depth_unit,
        )

        save_heatmap_pointclouds(
            results=results,
            depth_m=heat_depth_m,
            intr=heat_intr,
            output_dir=args.output_dir,
            max_depth_m=args.reconstruction_max_depth,
        )
    print("Predictions by symmetry:")
    print(predictions)
    print(f"Created {len(results)} result object(s).")

    reconstructed = sum(1 for result in results if result.reconstruction_3d is not None)

    if args.depth is not None:
        print(f"Created old/default 3D reconstruction for {reconstructed} object result(s).")

    if args.second_input is not None and args.second_depth is not None:
        print("Two-view asymmetric reconstruction inputs were provided.")

    if args.save_outputs:
        for result in results:
            result.save(args.output_dir)
        print(f"Saved 2D pipeline outputs to: {args.output_dir}")
    else:
        print("Saving disabled. Use --save-outputs to write 2D outputs to disk.")

    # ------------------------------------------------------------
    # Semantic 3D reconstruction for symmetric objects
    # ------------------------------------------------------------
    if args.symmetric_3d_semantic:
        depth_m, intr = prepare_depth_and_intrinsics(
            args.symmetric_depth,
            args.intrinsics,
            (args.height, args.width),
            args.depth_unit,
        )

        sym_out = args.output_dir / "symmetric_semantic_3d"
        sym_out.mkdir(parents=True, exist_ok=True)

        count = 0
        print("[SYMMETRIC 3D] Running symmetric image pipeline...")
        sym_predictions, sym_results = pipeline.run(args.symmetric_input, second_input_image=None)

        if args.save_heatmap_pointclouds:
            save_heatmap_pointclouds(
                results=sym_results,
                depth_m=depth_m,
                intr=intr,
                output_dir=args.output_dir / "symmetric_heatmaps",
                max_depth_m=args.reconstruction_max_depth,
            )
        print("Symmetric image predictions:")
        print(sym_predictions)

        if args.save_outputs:
            sym_2d_out = args.output_dir / "symmetric_view_2d"
            for result in sym_results:
                result.save(sym_2d_out)
            print(f"Saved symmetric-view 2D outputs to: {sym_2d_out}")

        for result in sym_results:
            if symmetry_for_name(result.object_name) != "symmetric":
                continue

            try:
                reconstruct_one_object(
                    object_name=result.object_name,
                    image_stem=result.image_path.stem,
                    image_path=result.image_path,
                    rgb=result.rgb_resized,
                    mask=result.sam_mask,
                    heatmap=result.final_heatmap,
                    depth_m=depth_m,
                    intr=intr,
                    output_dir=sym_out,
                    heatmap_threshold=args.threshold,
                    min_component_area=args.reconstruction_min_component_area,
                    max_depth_m=args.reconstruction_max_depth,
                    depth_window_m=args.depth_window,
                    center_depth_offset_ratio=args.center_depth_offset_ratio,
                    patch_source=args.patch_source,
                    patch_dilate_kernel=args.patch_dilate_kernel,
                    patch_dilate_iterations=args.patch_dilate_iterations,
                    patch_keep_percentile=args.patch_keep_percentile,
                )

                object_dir = sym_out / f"{result.image_path.stem}_{safe_name(result.object_name)}"

                if args.visualize_symmetric_3d:
                    visualize_symmetric_object_dir(
                        object_dir,
                        patch_id=1,
                        show=args.show_symmetric_3d,
                    )

                count += 1

            except Exception as exc:
                print(f"[WARNING] Semantic symmetric 3D failed for {result.object_name!r}: {exc}")

        print(f"Created semantic symmetric 3D reconstruction for {count} object(s).")
        print(f"Saved semantic symmetric 3D outputs to: {sym_out}")
        # ------------------------------------------------------------
    # Two-view 3D reconstruction for asymmetric objects
    # ------------------------------------------------------------
    if args.asymmetric_3d_twoview:
        print("[ASYMMETRIC 3D] Running second view pipeline...")

        predictions2, results2 = pipeline.run(args.second_input, second_input_image=None)

        print("Predictions view 2:")
        print(predictions2)

        depth1_m, asym_intr = prepare_asym_depth_and_intrinsics(
            args.depth,
            args.intrinsics,
            (args.height, args.width),
            args.depth_unit,
        )

        depth2_m, _ = prepare_asym_depth_and_intrinsics(
            args.second_depth,
            args.intrinsics,
            (args.height, args.width),
            args.depth_unit,
        )

        if args.save_heatmap_pointclouds:
            save_heatmap_pointclouds(
                results=results2,
                depth_m=depth2_m,
                intr=asym_intr,
                output_dir=args.output_dir / "asymmetric_view2_heatmaps",
                max_depth_m=args.reconstruction_max_depth,
            )

        pose1_raw = load_pose_matrix(args.pose)
        pose2_raw = load_pose_matrix(args.second_pose)

        if args.pose_convention == "world_to_camera":
            pose1 = np.linalg.inv(pose1_raw).astype(np.float32)
            pose2 = np.linalg.inv(pose2_raw).astype(np.float32)
        else:
            pose1 = pose1_raw.astype(np.float32)
            pose2 = pose2_raw.astype(np.float32)

        asym_out = args.output_dir / "asymmetric_twoview_3d"
        asym_out.mkdir(parents=True, exist_ok=True)

        print("[ASYMMETRIC 3D] Reconstructing view 1 asymmetric objects...")

        view1_objects = collect_asymmetric_view_results(
            pipeline_results=results,
            depth_m=depth1_m,
            intr=asym_intr,
            pose_cam_to_world=pose1,
            heatmap_threshold=args.threshold,
            min_component_area=args.reconstruction_min_component_area,
            max_depth_m=args.reconstruction_max_depth,
            depth_window_m=args.depth_window,
            patch_source=args.patch_source,
            patch_dilate_kernel=args.patch_dilate_kernel,
            patch_dilate_iterations=args.patch_dilate_iterations,
        )

        print("[ASYMMETRIC 3D] Reconstructing view 2 asymmetric objects...")

        view2_objects = collect_asymmetric_view_results(
            pipeline_results=results2,
            depth_m=depth2_m,
            intr=asym_intr,
            pose_cam_to_world=pose2,
            heatmap_threshold=args.threshold,
            min_component_area=args.reconstruction_min_component_area,
            max_depth_m=args.reconstruction_max_depth,
            depth_window_m=args.depth_window,
            patch_source=args.patch_source,
            patch_dilate_kernel=args.patch_dilate_kernel,
            patch_dilate_iterations=args.patch_dilate_iterations,
        )

        common_keys = sorted(set(view1_objects.keys()) & set(view2_objects.keys()))

        third_object_key = args.third_object_key.strip().lower()
        use_third_for_object = (
            args.third_input is not None
            and args.third_depth is not None
            and args.third_pose is not None
        )

        if use_third_for_object and third_object_key in common_keys:
            common_keys.remove(third_object_key)
            print(
                f"[ASYMMETRIC 3D] {third_object_key} will use the dedicated third view, "
                "not the normal view1/view2 fusion."
            )

        print(f"[ASYMMETRIC 3D] Common objects: {common_keys}")

        fused_root = asym_out / "fused_asymmetric_objects"
        fused_root.mkdir(parents=True, exist_ok=True)

        fused_count = 0

        for key in common_keys:
            item1 = max(view1_objects[key], key=lambda x: len(x["patch_world_xyz"]))
            item2 = max(view2_objects[key], key=lambda x: len(x["patch_world_xyz"]))

            try:
                print(f"[ASYMMETRIC 3D] Fusing {key}")

                obj_dir = save_fused_object(
                    output_dir=fused_root,
                    object_key=key,
                    view1_item=item1,
                    view2_item=item2,
                    image1_name=str(args.input),
                    image2_name=str(args.second_input),
                    pose1=pose1,
                    pose2=pose2,
                    intr=asym_intr,
                )

                if args.visualize_asymmetric_3d:
                    visualize_asymmetric_object_dir(
                        obj_dir,
                        show=args.show_asymmetric_3d,
                    )

                fused_count += 1

            except Exception as exc:
                print(f"[WARNING] Asymmetric fusion failed for {key!r}: {exc}")

        print(f"Created asymmetric two-view 3D fusion for {fused_count}/{len(common_keys)} object(s).")

        if use_third_for_object:
            print(
                f"[ASYMMETRIC 3D] Running third-view pipeline for {third_object_key!r} only..."
            )

            predictions3, results3 = pipeline.run(args.third_input, second_input_image=None)

            print("Predictions view 3:")
            print(predictions3)

            depth3_m, _ = prepare_asym_depth_and_intrinsics(
                args.third_depth,
                args.intrinsics,
                (args.height, args.width),
                args.depth_unit,
            )

            if args.save_heatmap_pointclouds:
                save_heatmap_pointclouds(
                    results=results3,
                    depth_m=depth3_m,
                    intr=asym_intr,
                    output_dir=args.output_dir / "asymmetric_view3_heatmaps",
                    max_depth_m=args.reconstruction_max_depth,
                )

            pose3_raw = load_pose_matrix(args.third_pose)

            if args.pose_convention == "world_to_camera":
                pose3 = np.linalg.inv(pose3_raw).astype(np.float32)
            else:
                pose3 = pose3_raw.astype(np.float32)

            third_results = [
                result
                for result in results3
                if safe_name(result.object_name) == third_object_key
            ]

            view3_objects = collect_asymmetric_view_results(
                pipeline_results=third_results,
                depth_m=depth3_m,
                intr=asym_intr,
                pose_cam_to_world=pose3,
                heatmap_threshold=args.threshold,
                min_component_area=args.reconstruction_min_component_area,
                max_depth_m=args.reconstruction_max_depth,
                depth_window_m=args.depth_window,
                patch_source=args.patch_source,
                patch_dilate_kernel=args.patch_dilate_kernel,
                patch_dilate_iterations=args.patch_dilate_iterations,
            )

            reference_items = view1_objects.get(third_object_key) or view2_objects.get(third_object_key)
            third_items = view3_objects.get(third_object_key)

            if not reference_items:
                print(
                    f"[WARNING] No reference object for {third_object_key!r} in view 1 or view 2."
                )
            elif not third_items:
                print(f"[WARNING] No third-view object for {third_object_key!r}.")
            else:
                item1 = max(reference_items, key=lambda x: len(x["patch_world_xyz"]))
                item3 = max(third_items, key=lambda x: len(x["patch_world_xyz"]))

                try:
                    print(f"[ASYMMETRIC 3D] Fusing {third_object_key} with dedicated third view")

                    obj_dir = save_fused_object(
                        output_dir=fused_root,
                        object_key=third_object_key,
                        view1_item=item1,
                        view2_item=item3,
                        image1_name=str(args.input),
                        image2_name=str(args.third_input),
                        pose1=pose1,
                        pose2=pose3,
                        intr=asym_intr,
                    )

                    if args.visualize_asymmetric_3d:
                        visualize_asymmetric_object_dir(
                            obj_dir,
                            show=args.show_asymmetric_3d,
                        )

                    print(f"Saved third-view fusion for {third_object_key!r} to: {obj_dir}")

                except Exception as exc:
                    print(f"[WARNING] Third-view fusion failed for {third_object_key!r}: {exc}")

        print(f"Saved asymmetric two-view outputs to: {asym_out}")

if __name__ == "__main__":
    main()
