from __future__ import annotations


from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from PIL import Image
# Workaround for broken optional MLX install in transformers.
# On Linux/CUDA, MLX is not needed, but transformers may still try to import it.
try:
    import transformers.utils.generic as hf_generic
    hf_generic._is_mlx_available = False
except Exception:
    pass

from transformers import (
    AutoProcessor,
    OwlViTForObjectDetection,
    OwlViTProcessor,
)

from src.object_catalog import (
    catalog_name_for_object,
    detector_queries,
    sam_prompt_for_object,
    split_by_symmetry,
)


try:
    from transformers import GroundingDinoForObjectDetection
except Exception:  # pragma: no cover - older transformers versions
    GroundingDinoForObjectDetection = None  # type: ignore[assignment]

DetectorBackend = Literal["grounding-dino", "owlvit"]


@dataclass(frozen=True)
class Detection:
    """One zero-shot detector hit, mapped back to the internal object catalog."""

    object_key: str
    object_name: str
    detector_prompt: str
    sam_prompt: str
    score: float
    box_xyxy: tuple[float, float, float, float]


class CatalogObjectDetector:
    """Open-vocabulary detector constrained to the local GraspNet/object catalog.

    Backends:
    - grounding-dino: stronger default for catalog/object-name detection.
    - owlvit: kept as a fallback when GroundingDINO dependencies/checkpoints are not available.

    Both backends return Detection objects mapped back to OBJECT_CATALOG keys, so downstream
    symmetry routing and SAM prompts stay unchanged.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
        backend: DetectorBackend = "grounding-dino",
    ) -> None:
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.backend = backend
        self.query_pairs: list[tuple[str, str]] = detector_queries()
        self.labels: list[str] = [prompt for _, prompt in self.query_pairs]

        if not self.labels:
            raise RuntimeError("No detector prompts were generated from OBJECT_CATALOG.")

        if backend == "grounding-dino":
            if GroundingDinoForObjectDetection is None:
                raise RuntimeError(
                    "GroundingDINO is not available in this transformers install. "
                    "Upgrade transformers or run with --detector-backend owlvit."
                )
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(self.device)
        elif backend == "owlvit":
            self.processor = OwlViTProcessor.from_pretrained(model_id)
            self.model = OwlViTForObjectDetection.from_pretrained(model_id).to(self.device)
        else:
            raise ValueError(f"Unsupported detector backend: {backend!r}")

        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        threshold: float = 0.10,
        top_k: int | None = None,
        deduplicate: bool = True,
        text_threshold: float = 0.20,
        label_batch_size: int = 20,
    ) -> list[Detection]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        label_batch_size = max(1, label_batch_size)
        detections: list[Detection] = []
        for start in range(0, len(self.query_pairs), label_batch_size):
            query_pairs = self.query_pairs[start : start + label_batch_size]
            if self.backend == "grounding-dino":
                detections.extend(self._predict_grounding_dino(image, threshold, text_threshold, query_pairs))
            else:
                detections.extend(self._predict_owlvit(image, threshold, query_pairs))

        detections.sort(key=lambda d: d.score, reverse=True)

        if deduplicate:
            best_by_object: dict[str, Detection] = {}
            for det in detections:
                best_by_object.setdefault(det.object_key, det)
            detections = sorted(best_by_object.values(), key=lambda d: d.score, reverse=True)

        if top_k is not None:
            detections = detections[:top_k]

        return detections

    def _predict_grounding_dino(
        self,
        image: Image.Image,
        box_threshold: float,
        text_threshold: float,
        query_pairs: list[tuple[str, str]],
    ) -> list[Detection]:
        # GroundingDINO works best with a period-separated caption and lowercase phrases.
        # Keep one-to-one phrase ordering so returned text labels can be mapped back.
        labels = [prompt for _, prompt in query_pairs]
        caption = ". ".join(labels) + "."
        inputs = self.processor(images=image, text=caption, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([(image.height, image.width)], device=self.device)

        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]

        boxes = processed.get("boxes", [])
        scores = processed.get("scores", [])
        text_labels = processed.get("text_labels", processed.get("labels", []))

        detections: list[Detection] = []
        for box, score, text_label in zip(boxes, scores, text_labels):
            label_text = str(text_label).strip().lower()
            match = self._match_grounding_label(label_text, query_pairs)
            if match is None:
                continue
            object_key, detector_prompt = match
            detections.append(
                Detection(
                    object_key=object_key,
                    object_name=catalog_name_for_object(object_key),
                    detector_prompt=detector_prompt,
                    sam_prompt=sam_prompt_for_object(object_key),
                    score=float(score.detach().cpu().item()),
                    box_xyxy=tuple(float(v) for v in box.detach().cpu().tolist()),
                )
            )
        return detections

    def _match_grounding_label(self, label_text: str, query_pairs: list[tuple[str, str]] | None = None) -> tuple[str, str] | None:
        # Exact phrase first, then substring fallback because GroundingDINO can return
        # phrases such as "small white bottle" for catalog label "white bottle".
        label_text = " ".join(label_text.replace("_", " ").replace("-", " ").split())
        normalized_pairs = [
            (key, prompt, " ".join(prompt.replace("_", " ").replace("-", " ").split()))
            for key, prompt in (query_pairs or self.query_pairs)
        ]
        for key, prompt, norm_prompt in normalized_pairs:
            if label_text == norm_prompt:
                return key, prompt
        for key, prompt, norm_prompt in normalized_pairs:
            if norm_prompt in label_text or label_text in norm_prompt:
                return key, prompt
        return None

    def _predict_owlvit(self, image: Image.Image, threshold: float, query_pairs: list[tuple[str, str]]) -> list[Detection]:
        labels = [prompt for _, prompt in query_pairs]
        text_labels = [labels]
        inputs = self.processor(text=text_labels, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([(image.height, image.width)], device=self.device)

        if hasattr(self.processor, "post_process_grounded_object_detection"):
            processed = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                threshold=threshold,
                target_sizes=target_sizes,
                text_labels=text_labels,
            )[0]
        elif hasattr(self.processor, "post_process_object_detection"):
            processed = self.processor.post_process_object_detection(
                outputs=outputs,
                threshold=threshold,
                target_sizes=target_sizes,
            )[0]
        else:
            raise RuntimeError("Unsupported OwlViTProcessor post-processing API.")

        boxes = processed.get("boxes", [])
        scores = processed.get("scores", [])

        if "labels" in processed:
            label_indices: Iterable[int] = [int(x.detach().cpu().item()) for x in processed["labels"]]
        elif "text_labels" in processed:
            prompt_to_idx = {prompt: idx for idx, prompt in enumerate(labels)}
            label_indices = [prompt_to_idx.get(str(label), -1) for label in processed["text_labels"]]
        else:
            raise RuntimeError(f"OWL-ViT result has neither 'labels' nor 'text_labels'. Keys: {list(processed.keys())}")

        detections: list[Detection] = []
        for box, score, label_idx in zip(boxes, scores, label_indices):
            if label_idx < 0 or label_idx >= len(query_pairs):
                continue
            object_key, detector_prompt = query_pairs[label_idx]
            detections.append(
                Detection(
                    object_key=object_key,
                    object_name=catalog_name_for_object(object_key),
                    detector_prompt=detector_prompt,
                    sam_prompt=sam_prompt_for_object(object_key),
                    score=float(score.detach().cpu().item()),
                    box_xyxy=tuple(float(v) for v in box.detach().cpu().tolist()),
                )
            )
        return detections

    def prediction_dictionary(self, detections: list[Detection]) -> dict[str, list[str]]:
        """Return catalog object names grouped by their symmetry label."""
        return split_by_symmetry(det.object_name for det in detections)


# Backward-compatible name used by older imports/tests.
ZeroShotObjectDetector = CatalogObjectDetector
