from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from .device_utils import recommended_sam_device, resolve_device
from .detector import CatalogObjectDetector, Detection
from .heatmap_model import HeatmapPredictor
from .io_utils import load_and_resize_rgb
from .reconstruction import (
    CameraIntrinsics,
    ReconstructionConfig,
    load_depth,
    reconstruct_symmetric_object,
    reconstruct_asymmetric_object,
)
from .object_catalog import symmetry_for_name, flexible_for_name, sam_prompt_for_object
from .postprocessing import apply_mask_to_rgb, stretch_heatmap_towards_edges
from .results import ObjectPipelineResult
from .sam_segmenter import Sam3Segmenter


@dataclass
class PipelineConfig:
    image_size: tuple[int, int] = (384, 640)
    device: str = "auto"
    detector_device: str | None = None
    sam_device: str | None = None
    heatmap_device: str | None = None

    detector_model_id: str = "IDEA-Research/grounding-dino-base"
    detector_backend: str = "grounding-dino"
    detector_threshold: float = 0.10
    detector_text_threshold: float = 0.20
    detector_label_batch_size: int = 20
    run_asymmetric_on_first_image: bool = True
    detector_top_k: int | None = None

    sam_model_id: str = "facebook/sam3"
    sam_score_threshold: float = 0.5
    sam_mask_threshold: float = 0.5

    heatmap_threshold: float | None = 0.4
    overlay_alpha: float = 0.45
    mask_background: str = "black"

    postprocess_edges: bool = False
    edge_shift_strength: float = 0.65
    edge_max_distance: float = 50.0
    heat_threshold: float = 0.1
    post_smooth_sigma: float = 1.6
    canny_low: int = 50
    canny_high: int = 150
    preserve_original: float = 0.25

    depth_path: Path | None = None
    second_depth_path: Path | None = None
    intrinsics_path: Path | None = None
    pose_path: Path | None = None
    second_pose_path: Path | None = None

    depth_unit: str = "m"
    reconstruction_min_component_area_px: int = 20
    reconstruction_max_depth_m: float = 3.0
    symmetry_axis_mode: str = "vertical"

    def __post_init__(self) -> None:
        self.device = resolve_device(self.device)

        detector_request = self.detector_device
        heatmap_request = self.heatmap_device
        sam_request = self.sam_device

        self.detector_device = (
            self.device
            if detector_request is None or detector_request.lower() == "auto"
            else resolve_device(detector_request)
        )
        self.heatmap_device = (
            self.device
            if heatmap_request is None or heatmap_request.lower() == "auto"
            else resolve_device(heatmap_request)
        )
        self.sam_device = (
            recommended_sam_device(self.device)
            if sam_request is None or sam_request.lower() == "auto"
            else resolve_device(sam_request)
        )


class SymmetryAwareSamPipeline:
    def __init__(self, heatmap_model_path: Path, config: PipelineConfig) -> None:
        self.config = config

        self.detector = CatalogObjectDetector(
            model_id=config.detector_model_id,
            device=config.detector_device,
            backend=config.detector_backend,
        )

        self.sam = Sam3Segmenter(
            model_id=config.sam_model_id,
            device=config.sam_device,
        )

        self.heatmap = HeatmapPredictor(
            heatmap_model_path,
            config.heatmap_device,
        )

        self._intrinsics = (
            CameraIntrinsics.from_json(config.intrinsics_path)
            if config.intrinsics_path
            else None
        )

        self._pose1 = self._load_pose(config.pose_path) if config.pose_path else None
        self._pose2 = self._load_pose(config.second_pose_path) if config.second_pose_path else None

    def _load_pose(self, path: Path) -> np.ndarray:
        vals = np.loadtxt(path).astype(np.float32).reshape(-1)

        if vals.size < 6:
            raise ValueError(f"Pose file must contain at least 6 values: {path}")

        tx, ty, tz, rx, ry, rz = vals[:6]

        rx_mat = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(rx), -np.sin(rx)],
                [0.0, np.sin(rx), np.cos(rx)],
            ],
            dtype=np.float32,
        )

        ry_mat = np.array(
            [
                [np.cos(ry), 0.0, np.sin(ry)],
                [0.0, 1.0, 0.0],
                [-np.sin(ry), 0.0, np.cos(ry)],
            ],
            dtype=np.float32,
        )

        rz_mat = np.array(
            [
                [np.cos(rz), -np.sin(rz), 0.0],
                [np.sin(rz), np.cos(rz), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        rotation = rz_mat @ ry_mat @ rx_mat

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)

        return transform

    def detect(self, image: Image.Image) -> tuple[list[Detection], dict[str, list[str]]]:
        detections = self.detector.predict(
            image,
            threshold=self.config.detector_threshold,
            top_k=self.config.detector_top_k,
            text_threshold=self.config.detector_text_threshold,
            label_batch_size=self.config.detector_label_batch_size,
        )

        return detections, self.detector.prediction_dictionary(detections)

    def run_for_prompt(self, image_path: Path, object_name: str) -> ObjectPipelineResult | None:
        cfg = self.config

        rgb_pil = load_and_resize_rgb(image_path, cfg.image_size)
        rgb_np = np.asarray(rgb_pil)

        sam_prompt = sam_prompt_for_object(object_name)
        sam_prompts = [sam_prompt]

        if object_name == "hand cream tube":
            sam_prompts = [
                "hand cream tube",
                "white hand cream tube",
                "white tube with black label",
                "cream tube",
                "white cosmetic tube",
                "white tube",
            ]

        sam_mask = None
        sam_raw_output = {}
        used_sam_prompt = sam_prompt

        for candidate_prompt in sam_prompts:
            try:
                sam_mask, sam_raw_output = self.sam.segment(
                    image=rgb_pil,
                    text_prompt=candidate_prompt,
                    score_threshold=cfg.sam_score_threshold,
                    mask_threshold=cfg.sam_mask_threshold,
                )
                used_sam_prompt = candidate_prompt
                break

            except RuntimeError as exc:
                message = str(exc)

                if "no masks" in message.lower() or "prompt" in message.lower():
                    print(
                        f"[WARNING] SAM3 skipped prompt {candidate_prompt!r} "
                        f"for object {object_name!r}: {message}"
                    )
                    continue

                raise

            except Exception as exc:
                print(
                    f"[WARNING] SAM3 failed for prompt {candidate_prompt!r} "
                    f"for object {object_name!r}: {exc}"
                )
                continue

        if sam_mask is None:
            print(
                f"[WARNING] SAM3 returned empty mask for prompts {sam_prompts!r} "
                f"for object {object_name!r}. Skipping."
            )
            return None

        if used_sam_prompt != sam_prompt:
            print(
                f"[INFO] Used fallback SAM prompt {used_sam_prompt!r} "
                f"for object {object_name!r}."
            )

        masked_rgb_np = apply_mask_to_rgb(
            rgb_np,
            sam_mask,
            background=cfg.mask_background,
        )

        raw_heatmap = self.heatmap.predict(Image.fromarray(masked_rgb_np))

        postprocessed = None
        edge_map = None

        if cfg.postprocess_edges:
            postprocessed, edge_map = stretch_heatmap_towards_edges(
                heatmap=raw_heatmap,
                object_mask=sam_mask,
                rgb_image=masked_rgb_np,
                strength=cfg.edge_shift_strength,
                heat_threshold=cfg.heat_threshold,
                max_distance=cfg.edge_max_distance,
                smooth_sigma=cfg.post_smooth_sigma,
                canny_low=cfg.canny_low,
                canny_high=cfg.canny_high,
                preserve_original=cfg.preserve_original,
            )

        reconstruction_3d = None

        if (
            symmetry_for_name(object_name) == "symmetric"
            and cfg.depth_path is not None
            and self._intrinsics is not None
        ):
            depth_m = load_depth(
                cfg.depth_path,
                cfg.image_size,
                depth_unit=cfg.depth_unit,
            )

            recon_cfg = ReconstructionConfig(
                heatmap_threshold=cfg.heatmap_threshold
                if cfg.heatmap_threshold is not None
                else 0.4,
                min_component_area_px=cfg.reconstruction_min_component_area_px,
                max_depth_m=cfg.reconstruction_max_depth_m,
                depth_unit=cfg.depth_unit,
                symmetry_axis_mode=cfg.symmetry_axis_mode,
            )

            reconstruction_3d = reconstruct_symmetric_object(
                image_path=image_path,
                object_name=object_name,
                rgb=rgb_np,
                object_mask=sam_mask,
                heatmap=postprocessed if postprocessed is not None else raw_heatmap,
                depth_m=depth_m,
                intrinsics=self._intrinsics,
                config=recon_cfg,
                object_flexible=flexible_for_name(object_name),
            )

        return ObjectPipelineResult(
            image_path=image_path,
            object_name=object_name,
            sam_prompt=used_sam_prompt,
            symmetry=symmetry_for_name(object_name),
            flexible=flexible_for_name(object_name),
            rgb_resized=rgb_np,
            sam_mask=sam_mask,
            masked_rgb=masked_rgb_np,
            raw_heatmap=raw_heatmap,
            postprocessed_heatmap=postprocessed,
            edge_map=edge_map,
            threshold=cfg.heatmap_threshold,
            overlay_alpha=cfg.overlay_alpha,
            sam_raw_output=sam_raw_output,
            reconstruction_3d=reconstruction_3d,
        )

    def _can_run_asymmetric_3d(self, second_input_image: Path | None) -> bool:
        cfg = self.config

        return (
            second_input_image is not None
            and cfg.depth_path is not None
            and cfg.second_depth_path is not None
            and self._intrinsics is not None
            and self._pose1 is not None
            and self._pose2 is not None
        )

    @staticmethod
    def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        a = mask_a.astype(bool)
        b = mask_b.astype(bool)
        union = np.logical_or(a, b).sum()
        if union == 0:
            return 0.0
        return float(np.logical_and(a, b).sum() / union)

    def _append_if_not_duplicate(
        self,
        results: list[ObjectPipelineResult],
        result: ObjectPipelineResult,
        *,
        iou_threshold: float = 0.72,
    ) -> bool:
        preferred_duplicate_names = {"hand cream tube"}

        for idx, existing in enumerate(results):
            if existing.image_path != result.image_path:
                continue
            iou = self._mask_iou(existing.sam_mask, result.sam_mask)
            if iou >= iou_threshold:
                if (
                    result.object_name in preferred_duplicate_names
                    and existing.object_name not in preferred_duplicate_names
                ):
                    print(
                        f"[INFO] Replaced duplicate mask {existing.object_name!r} "
                        f"with preferred object {result.object_name!r}; IoU={iou:.2f}."
                    )
                    results[idx] = result
                    return True

                print(
                    f"[INFO] Skipped duplicate mask for {result.object_name!r}; "
                    f"overlaps {existing.object_name!r} with IoU={iou:.2f}."
                )
                return False

        results.append(result)
        return True

    def run(
        self,
        input_image: Path,
        second_input_image: Path | None = None,
    ) -> tuple[dict[str, list[str]], list[ObjectPipelineResult]]:
        cfg = self.config

        first_image = load_and_resize_rgb(input_image, cfg.image_size)
        _, predictions = self.detect(first_image)

        if (
            "hand cream tube" not in predictions["asymmetric"]
            and "hand cream tube" not in predictions["symmetric"]
        ):
            predictions["symmetric"].append("hand cream tube")
            print("[INFO] Added forced symmetric prompt for 'hand cream tube'.")

        results: list[ObjectPipelineResult] = []

        for object_name in predictions["symmetric"]:
            result = self.run_for_prompt(input_image, object_name)

            if result is not None:
                self._append_if_not_duplicate(results, result)
            else:
                print(f"[WARNING] Skipped symmetric object {object_name!r}.")

        if predictions["asymmetric"]:
            if not self._can_run_asymmetric_3d(second_input_image):
                print(
                    "[WARNING] Asymmetric objects detected, but two-view 3D inputs are incomplete. "
                    "Asymmetric objects will be processed as normal 2D results only."
                )

                asymmetric_image_paths: list[Path] = []

                if cfg.run_asymmetric_on_first_image:
                    asymmetric_image_paths.append(input_image)

                if second_input_image is not None:
                    asymmetric_image_paths.append(second_input_image)

                if not asymmetric_image_paths:
                    raise RuntimeError(
                        "Asymmetric objects were detected, but no image is configured for the asymmetric stage."
                    )

                for image_path in asymmetric_image_paths:
                    for object_name in predictions["asymmetric"]:
                        result = self.run_for_prompt(image_path, object_name)

                        if result is not None:
                            self._append_if_not_duplicate(results, result)
                        else:
                            print(
                                f"[WARNING] Skipped asymmetric object {object_name!r} "
                                f"on image {image_path}."
                            )

            else:
                depth1 = load_depth(
                    cfg.depth_path,
                    cfg.image_size,
                    depth_unit=cfg.depth_unit,
                )

                depth2 = load_depth(
                    cfg.second_depth_path,
                    cfg.image_size,
                    depth_unit=cfg.depth_unit,
                )

                for object_name in predictions["asymmetric"]:
                    print(f"[INFO] Running two-view asymmetric reconstruction for {object_name!r}")

                    result1 = self.run_for_prompt(input_image, object_name)
                    result2 = self.run_for_prompt(second_input_image, object_name)

                    if result1 is None or result2 is None:
                        print(
                            f"[WARNING] Could not segment asymmetric object {object_name!r} "
                            "in both views. Skipping fused reconstruction."
                        )
                        continue

                    heatmap1 = (
                        result1.postprocessed_heatmap
                        if result1.postprocessed_heatmap is not None
                        else result1.raw_heatmap
                    )

                    heatmap2 = (
                        result2.postprocessed_heatmap
                        if result2.postprocessed_heatmap is not None
                        else result2.raw_heatmap
                    )

                    recon_cfg = ReconstructionConfig(
                        heatmap_threshold=cfg.heatmap_threshold
                        if cfg.heatmap_threshold is not None
                        else 0.4,
                        min_component_area_px=cfg.reconstruction_min_component_area_px,
                        max_depth_m=cfg.reconstruction_max_depth_m,
                        depth_unit=cfg.depth_unit,
                        symmetry_axis_mode=cfg.symmetry_axis_mode,
                    )

                    fused_reconstruction = reconstruct_asymmetric_object(
                        image_path_1=input_image,
                        image_path_2=second_input_image,
                        object_name=object_name,
                        rgb1=result1.rgb_resized,
                        rgb2=result2.rgb_resized,
                        object_mask1=result1.sam_mask,
                        object_mask2=result2.sam_mask,
                        heatmap1=heatmap1,
                        heatmap2=heatmap2,
                        depth_m1=depth1,
                        depth_m2=depth2,
                        intrinsics=self._intrinsics,
                        pose1=self._pose1,
                        pose2=self._pose2,
                        config=recon_cfg,
                        object_flexible=flexible_for_name(object_name),
                    )

                    result1.reconstruction_3d = fused_reconstruction
                    self._append_if_not_duplicate(results, result1)

        return predictions, results
