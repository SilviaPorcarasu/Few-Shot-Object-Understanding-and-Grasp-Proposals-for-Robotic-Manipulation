from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x - mean) / std


def preprocess_pil_rgb(img: Image.Image, device: str) -> torch.Tensor:
    x = TF.to_tensor(img)
    x = normalize_imagenet(x)
    return x.unsqueeze(0).to(device)


class HeatmapPredictor:
    def __init__(self, model_path: Path, device: str) -> None:
        if not model_path.exists():
            raise RuntimeError(f"Heatmap model not found: {model_path}")
        self.device = device
        self.model = torch.jit.load(str(model_path), map_location=device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: Image.Image) -> np.ndarray:
        x = preprocess_pil_rgb(image, self.device)
        logits = self.model(x)
        pred = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        return np.clip(pred, 0.0, 1.0)
