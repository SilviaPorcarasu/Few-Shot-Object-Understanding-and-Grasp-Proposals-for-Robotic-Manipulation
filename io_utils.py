from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMG_EXTS:
            raise RuntimeError(f"Unsupported image file: {input_path}")
        return [input_path]
    if input_path.is_dir():
        images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not images:
            raise RuntimeError(f"No images found in folder: {input_path}")
        return images
    raise RuntimeError(f"Input path does not exist: {input_path}")


def load_and_resize_rgb(image_path: Path, image_size: Tuple[int, int]) -> Image.Image:
    h, w = image_size
    img = Image.open(image_path).convert("RGB")
    return img.resize((w, h), resample=Image.BILINEAR)
