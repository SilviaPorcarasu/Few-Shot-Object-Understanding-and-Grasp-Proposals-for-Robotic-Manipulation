from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def colorize_heatmap(hm: np.ndarray) -> np.ndarray:
    hm = np.clip(hm, 0.0, 1.0)
    rgba = plt.get_cmap("jet")(hm)
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay_heatmap_on_rgb(rgb: np.ndarray, hm: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = colorize_heatmap(hm).astype(np.float32) / 255.0
    rgb_f = rgb.astype(np.float32) / 255.0
    out = (1.0 - alpha) * rgb_f + alpha * heat
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
