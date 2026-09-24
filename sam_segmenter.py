from __future__ import annotations

from typing import Tuple
import numpy as np
import torch
from PIL import Image


def masks_to_binary_union(masks) -> np.ndarray:
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    if isinstance(masks, list):
        arrs = [to_numpy(m) for m in masks]
        if not arrs:
            raise RuntimeError("SAM3 returned an empty mask list.")
        stacked = np.stack(arrs, axis=0)
    else:
        stacked = to_numpy(masks)

    if stacked.ndim == 2:
        union = stacked > 0
    elif stacked.ndim == 3:
        union = np.any(stacked > 0, axis=0)
    else:
        raise RuntimeError(f"Unsupported SAM3 mask shape: {stacked.shape}")
    return union.astype(np.uint8)


class Sam3Segmenter:
    def __init__(self, model_id: str = "facebook/sam3", device: str = "cuda") -> None:
        from transformers import Sam3Model, Sam3Processor

        self.device = device
        self.model = Sam3Model.from_pretrained(model_id).to(device)
        self.model.eval()
        self.processor = Sam3Processor.from_pretrained(model_id)

    @torch.no_grad()
    def segment(
        self,
        image: Image.Image,
        text_prompt: str,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> Tuple[np.ndarray, dict]:
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        if "masks" not in results:
            raise RuntimeError(f"SAM3 result does not contain 'masks'. Keys: {list(results.keys())}")
        if len(results["masks"]) == 0:
            raise RuntimeError(f"SAM3 returned no masks for prompt: {text_prompt!r}")

        return masks_to_binary_union(results["masks"]), {
            "boxes": results.get("boxes"),
            "scores": results.get("scores"),
            "labels": results.get("labels"),
        }
