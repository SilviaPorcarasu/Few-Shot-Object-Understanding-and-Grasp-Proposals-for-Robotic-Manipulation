from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any
import numpy as np
from PIL import Image

from .visualization import overlay_heatmap_on_rgb
from .postprocessing import threshold_heatmap
from .reconstruction import ObjectReconstruction3D


@dataclass
class ObjectPipelineResult:
    image_path: Path
    object_name: str
    sam_prompt: str
    symmetry: str
    flexible: bool
    rgb_resized: np.ndarray
    sam_mask: np.ndarray
    masked_rgb: np.ndarray
    raw_heatmap: np.ndarray
    postprocessed_heatmap: np.ndarray | None = None
    edge_map: np.ndarray | None = None
    threshold: float | None = None
    overlay_alpha: float = 0.45
    sam_raw_output: dict[str, Any] = field(default_factory=dict)
    reconstruction_3d: ObjectReconstruction3D | None = None

    @property
    def final_heatmap(self) -> np.ndarray:
        return self.postprocessed_heatmap if self.postprocessed_heatmap is not None else self.raw_heatmap

    @property
    def thresholded_heatmap(self) -> np.ndarray | None:
        return threshold_heatmap(self.final_heatmap, self.threshold)

    @property
    def raw_overlay(self) -> np.ndarray:
        return overlay_heatmap_on_rgb(self.masked_rgb, self.raw_heatmap, self.overlay_alpha)

    @property
    def final_overlay(self) -> np.ndarray:
        return overlay_heatmap_on_rgb(self.masked_rgb, self.final_heatmap, self.overlay_alpha)

    @property
    def thresholded_overlay(self) -> np.ndarray | None:
        thr = self.thresholded_heatmap
        if thr is None:
            return None
        return overlay_heatmap_on_rgb(self.masked_rgb, thr, self.overlay_alpha)

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_obj = self.object_name.replace(" ", "_").replace("/", "_")
        stem = f"{self.image_path.stem}_{safe_obj}"

        metadata = {
            "object_name": self.object_name,
            "sam_prompt": self.sam_prompt,
            "symmetry": self.symmetry,
            "flexible": bool(self.flexible),
            "object_flexible": bool(self.flexible),
            "image_path": str(self.image_path),
        }
        (output_dir / f"{stem}_pipeline_metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        Image.fromarray(self.rgb_resized).save(output_dir / f"{stem}_rgb_resized.png")
        Image.fromarray((self.sam_mask * 255).astype(np.uint8)).save(output_dir / f"{stem}_sam3_mask.png")
        Image.fromarray(self.masked_rgb).save(output_dir / f"{stem}_rgb_masked.png")
        Image.fromarray((self.raw_heatmap * 255).astype(np.uint8)).save(output_dir / f"{stem}_pred_raw_gray.png")
        Image.fromarray(self.raw_overlay).save(output_dir / f"{stem}_pred_raw_overlay.png")
        Image.fromarray((self.final_heatmap * 255).astype(np.uint8)).save(output_dir / f"{stem}_pred_final_gray.png")
        Image.fromarray(self.final_overlay).save(output_dir / f"{stem}_pred_final_overlay.png")

        if self.thresholded_heatmap is not None:
            Image.fromarray((self.thresholded_heatmap * 255).astype(np.uint8)).save(output_dir / f"{stem}_pred_thresholded.png")
            Image.fromarray(self.thresholded_overlay).save(output_dir / f"{stem}_pred_thresholded_overlay.png")
        if self.edge_map is not None:
            Image.fromarray((self.edge_map * 255).astype(np.uint8)).save(output_dir / f"{stem}_sam3_edges.png")
        if self.reconstruction_3d is not None:
            self.reconstruction_3d.save(output_dir, stem)
