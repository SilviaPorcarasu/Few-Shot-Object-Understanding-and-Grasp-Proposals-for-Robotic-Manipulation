from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


def apply_mask_to_rgb(rgb: np.ndarray, mask: np.ndarray, background: str = "black") -> np.ndarray:
    if background not in {"black", "white"}:
        raise ValueError("background must be 'black' or 'white'")
    out = rgb.copy()
    out[mask == 0] = 0 if background == "black" else 255
    return out


def canny_edges_from_rgb(rgb: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return (cv2.Canny(gray, low, high) > 0).astype(np.uint8)


def threshold_heatmap(heatmap: np.ndarray, threshold: float | None) -> np.ndarray | None:
    if threshold is None:
        return None
    return np.where(heatmap >= threshold, heatmap, 0.0).astype(np.float32)


def stretch_heatmap_towards_edges(
    heatmap: np.ndarray,
    object_mask: np.ndarray,
    rgb_image: np.ndarray,
    strength: float = 0.35,
    heat_threshold: float = 0.03,
    max_distance: float = 80.0,
    smooth_sigma: float = 1.5,
    canny_low: int = 50,
    canny_high: int = 150,
    preserve_original: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    hm = np.clip(heatmap, 0.0, 1.0).astype(np.float32)
    mask = (object_mask > 0).astype(np.uint8)
    edge_map = canny_edges_from_rgb(rgb_image, low=canny_low, high=canny_high) * mask
    if edge_map.sum() == 0:
        return hm * mask.astype(np.float32), edge_map

    edge_points = np.argwhere(edge_map > 0)
    active = (hm >= heat_threshold) & (mask > 0)
    ys, xs = np.where(active)
    if len(ys) == 0:
        return hm * mask.astype(np.float32), edge_map

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(edge_points)
        dists, idxs = tree.query(np.stack([ys, xs], axis=1), k=1)
        nearest = edge_points[idxs]
        edge_ys = nearest[:, 0].astype(np.float32)
        edge_xs = nearest[:, 1].astype(np.float32)
        dists = dists.astype(np.float32)
    except Exception:
        edge_ys = np.empty_like(ys, dtype=np.float32)
        edge_xs = np.empty_like(xs, dtype=np.float32)
        dists = np.empty(len(ys), dtype=np.float32)
        chunk = 2048
        points = np.stack([ys, xs], axis=1).astype(np.float32)
        edges = edge_points.astype(np.float32)
        for start in range(0, len(points), chunk):
            end = min(start + chunk, len(points))
            p = points[start:end]
            diff = p[:, None, :] - edges[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            idx = np.argmin(d2, axis=1)
            nearest = edge_points[idx]
            edge_ys[start:end] = nearest[:, 0]
            edge_xs[start:end] = nearest[:, 1]
            dists[start:end] = np.sqrt(d2[np.arange(end - start), idx])

    dist_factor = np.clip(1.0 - (dists / max_distance), 0.0, 1.0)
    local_strength = strength * dist_factor
    new_ys = ys.astype(np.float32) + local_strength * (edge_ys - ys.astype(np.float32))
    new_xs = xs.astype(np.float32) + local_strength * (edge_xs - xs.astype(np.float32))

    h, w = hm.shape
    values = hm[ys, xs]
    warped_sum = np.zeros_like(hm, dtype=np.float32)
    warped_weight = np.zeros_like(hm, dtype=np.float32)
    x0 = np.floor(new_xs).astype(np.int32)
    y0 = np.floor(new_ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = new_xs - x0
    wy = new_ys - y0

    for yy, xx, ww in [
        (y0, x0, (1.0 - wx) * (1.0 - wy)),
        (y0, x1, wx * (1.0 - wy)),
        (y1, x0, (1.0 - wx) * wy),
        (y1, x1, wx * wy),
    ]:
        valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        np.add.at(warped_sum, (yy[valid], xx[valid]), values[valid] * ww[valid])
        np.add.at(warped_weight, (yy[valid], xx[valid]), ww[valid])

    warped = warped_sum / np.maximum(warped_weight, 1e-6)
    blended = np.maximum(preserve_original * hm, warped) * mask.astype(np.float32)
    if smooth_sigma > 0:
        blended = cv2.GaussianBlur(blended, ksize=(0, 0), sigmaX=smooth_sigma)
    return np.clip(blended * mask.astype(np.float32), 0.0, 1.0), edge_map
