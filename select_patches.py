"""
STAGE 1: Split the grasp zone into smaller sub-zones (patches).

When a heatmap is available, every cloud point already has a per-pixel
ML affordance value. The patches are still seeded by farthest-point
sampling so they cover the surface evenly, but afterwards they are
sorted by their MEAN heatmap intensity — the brightest patches go
first. This way every later stage starts processing from the region
the ML model considers most graspable.

Input:  grasp_patch_points (N, 3) — points in the grasp zone
        grasp_heatmap_values (N,) — per-point heatmap value (0..1) or
                                    None when the heatmap is disabled
        grasp_pixels (N, 2)       — optional object pixels for a generic
                                    visible-edge prior
Output: list of Patch (sorted by descending mean_heatmap)
"""

from __future__ import annotations

import numpy as np

from grasp_config import PATCH_MIN_POINTS, PATCH_RADIUS_M, SEED_COUNT
from grasp_helpers import farthest_point_sampling
from grasp_types import Patch


def build_global_patch(
    grasp_points: np.ndarray,
    grasp_heatmap_values: np.ndarray | None = None,
    grasp_pixels: np.ndarray | None = None,
) -> Patch:
    """
    Build a single patch that spans the full object search space.

    This is the simplest analytical baseline: one PCA frame on the whole
    object, followed by the same contact generator and scorer used by the
    richer local-patch pipeline.
    """
    centroid = grasp_points.mean(axis=0)
    seed_index = int(np.argmin(np.linalg.norm(grasp_points - centroid, axis=1)))

    mean_hm = float(np.mean(grasp_heatmap_values)) if grasp_heatmap_values is not None else 0.0
    visible_edge_score = 0.0
    if grasp_pixels is not None and len(grasp_pixels) > 0:
        point_edge_scores = _visible_edge_scores_from_pixels(grasp_pixels)
        visible_edge_score = float(np.quantile(point_edge_scores, 0.75))

    return Patch(
        patch_id=0,
        seed_index=seed_index,
        seed_point=grasp_points[seed_index],
        point_indices=np.arange(len(grasp_points), dtype=np.int32),
        mean_heatmap=mean_hm,
        visible_edge_score=visible_edge_score,
    )


def _build_patches_for_radius(
    grasp_patch_points: np.ndarray,
    grasp_heatmap_values: np.ndarray | None,
    grasp_pixels: np.ndarray | None,
    seed_indices: np.ndarray,
    radius_m: float,
) -> list[Patch]:
    """
    Build patches for one chosen neighbourhood radius.

    We keep this helper tiny and explicit so the fallback logic below stays
    easy to follow.
    """
    patches: list[Patch] = []
    object_bbox_px: tuple[float, float, float, float] | None = None
    point_edge_scores: np.ndarray | None = None
    if grasp_pixels is not None and len(grasp_pixels) > 0:
        u = grasp_pixels[:, 0].astype(np.float64)
        v = grasp_pixels[:, 1].astype(np.float64)
        object_bbox_px = (float(np.min(u)), float(np.max(u)),
                          float(np.min(v)), float(np.max(v)))
        point_edge_scores = _visible_edge_scores_from_pixels(grasp_pixels)

    for patch_id, seed_idx in enumerate(seed_indices):
        seed_point = grasp_patch_points[int(seed_idx)]
        distances = np.linalg.norm(grasp_patch_points - seed_point, axis=1)
        nearby = np.flatnonzero(distances <= radius_m)
        if len(nearby) < PATCH_MIN_POINTS:
            continue

        if grasp_heatmap_values is not None:
            mean_hm = float(np.mean(grasp_heatmap_values[nearby]))
        else:
            mean_hm = 0.0

        visible_edge_score = 0.0
        if object_bbox_px is not None and point_edge_scores is not None:
            patch_pixels = grasp_pixels[nearby]
            bbox_score = _visible_edge_score_from_pixels(
                patch_pixels, object_bbox_px,
            )
            local_scores = point_edge_scores[nearby]
            hull_score = float(np.quantile(local_scores, 0.75))
            visible_edge_score = max(bbox_score, hull_score)

        patches.append(Patch(
            patch_id=patch_id,
            seed_index=int(seed_idx),
            seed_point=seed_point,
            point_indices=nearby,
            mean_heatmap=mean_hm,
            visible_edge_score=visible_edge_score,
        ))
    return patches


def _visible_edge_score_from_pixels(
    patch_pixels: np.ndarray,
    object_bbox_px: tuple[float, float, float, float],
) -> float:
    """
    Generic prior: prefer patches close to the observed object boundary.

    We do not modify the upstream heatmap. We only score where the patch
    lies inside the observed object extent. The score is high when the
    patch centroid is close to any side of the visible silhouette and
    low when it sits in the middle of the visible face.
    """
    if len(patch_pixels) == 0:
        return 0.0

    u_min, u_max, v_min, v_max = object_bbox_px
    width = max(u_max - u_min, 1.0)
    height = max(v_max - v_min, 1.0)

    patch_u = float(np.mean(patch_pixels[:, 0]))
    patch_v = float(np.mean(patch_pixels[:, 1]))

    x_norm = float(np.clip((patch_u - u_min) / width, 0.0, 1.0))
    y_norm = float(np.clip((patch_v - v_min) / height, 0.0, 1.0))

    nearest_edge = min(x_norm, 1.0 - x_norm, y_norm, 1.0 - y_norm)
    return float(np.clip(1.0 - 2.0 * nearest_edge, 0.0, 1.0))


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """
    Monotonic-chain convex hull for 2D points.

    Returns hull vertices in counter-clockwise order. Duplicates are removed.
    """
    if len(points) <= 1:
        return points.copy()

    pts = np.unique(points.astype(np.float64), axis=0)
    if len(pts) <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        oa = a - o
        ob = b - o
        return float(oa[0] * ob[1] - oa[1] * ob[0])

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def _point_to_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance from one 2D point to one 2D segment."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(((point - a) @ ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def _visible_edge_scores_from_pixels(grasp_pixels: np.ndarray) -> np.ndarray:
    """
    Generic visible-edge prior for every observed object point.

    The score is high near the OUTER silhouette of the observed object and
    low in the interior. Unlike a plain bounding-box heuristic, this uses
    the 2D contour implied by the projected point cloud, so lateral edges
    and slanted silhouettes count too.
    """
    if len(grasp_pixels) == 0:
        return np.zeros(0, dtype=np.float64)

    pixels_2d = grasp_pixels.astype(np.float64)
    hull = _convex_hull_2d(pixels_2d)
    if len(hull) < 2:
        return np.ones(len(grasp_pixels), dtype=np.float64)

    u = pixels_2d[:, 0]
    v = pixels_2d[:, 1]
    bbox_diag = max(float(np.hypot(np.max(u) - np.min(u), np.max(v) - np.min(v))), 1.0)
    edge_band_px = max(6.0, 0.10 * bbox_diag)

    distances = np.full(len(pixels_2d), np.inf, dtype=np.float64)
    for i in range(len(hull)):
        a = hull[i]
        b = hull[(i + 1) % len(hull)]
        seg_dist = np.array(
            [_point_to_segment_distance(p, a, b) for p in pixels_2d],
            dtype=np.float64,
        )
        distances = np.minimum(distances, seg_dist)

    scores = 1.0 - np.clip(distances / edge_band_px, 0.0, 1.0)
    return scores.astype(np.float64, copy=False)


def select_patches(
    grasp_patch_points: np.ndarray,
    grasp_heatmap_values: np.ndarray | None = None,
    grasp_pixels: np.ndarray | None = None,
) -> list[Patch]:
    """
    Spread seed points across the grasp zone and collect neighbours.

    Returns a list of valid patches (those with enough points), sorted
    by mean heatmap intensity descending when a heatmap is available.
    """
    # Step 1: pick seed points as far apart from each other as possible
    seed_indices = farthest_point_sampling(grasp_patch_points, SEED_COUNT)

    # Step 2: start with the configured radius. If a sparse cloud would
    # otherwise produce zero patches, relax only OUR local grouping rule.
    # This does not alter the upstream point cloud; it only avoids a hard
    # failure on valid but lower-density reconstructions.
    radius_candidates_m = (
        PATCH_RADIUS_M,
        PATCH_RADIUS_M * 1.5,
        PATCH_RADIUS_M * 2.0,
    )
    patches: list[Patch] = []
    for radius_m in radius_candidates_m:
        patches = _build_patches_for_radius(
            grasp_patch_points, grasp_heatmap_values, grasp_pixels,
            seed_indices, radius_m,
        )
        if patches:
            break

    # Step 5: brightest patches first — this is where the ML model
    # thinks the object is most graspable, so we want every downstream
    # stage to start its work from there.
    if grasp_heatmap_values is not None:
        patches.sort(key=lambda p: p.mean_heatmap, reverse=True)

    return patches
