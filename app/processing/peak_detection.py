"""Peak detection and non-maximum suppression for plant localization."""

from typing import List, Dict
import numpy as np
from scipy.ndimage import maximum_filter, label, find_objects
from scipy.spatial import KDTree


def get_peaks(
    density_map: np.ndarray,
    threshold: float = 0.001,
    neighborhood_size: int = 7,
) -> List[tuple]:
    """Find local maxima in a density map.

    Returns list of (row, col) coordinates of detected peaks.
    """
    neighborhood = maximum_filter(np.asarray(density_map, dtype=np.float32), size=neighborhood_size)
    peaks = (np.asarray(density_map, dtype=np.float32) == neighborhood) & (density_map > threshold)
    labeled, _ = label(peaks)
    slices = find_objects(labeled)
    points = [
        (int((dy.start + dy.stop - 1) / 2), int((dx.start + dx.stop - 1) / 2))
        for dy, dx in slices
    ]
    return points


def nms_merge_points(
    points: List[Dict],
    radius: float = 25.0,
) -> List[Dict]:
    """Merge nearby detections using non-maximum suppression.

    Each point is a dict with 'x', 'y', 'base_sigma' keys.
    Points within `radius` pixels are averaged into one detection.
    """
    if not points:
        return []
    pts = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
    tree = KDTree(pts)
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    merged = []
    for i in range(n):
        if visited[i]:
            continue
        idxs = tree.query_ball_point(pts[i], r=radius)
        xs = [points[j]["x"] for j in idxs]
        ys = [points[j]["y"] for j in idxs]
        sigs = [points[j]["base_sigma"] for j in idxs]
        merged.append({
            "x": float(np.mean(xs)),
            "y": float(np.mean(ys)),
            "base_sigma": float(np.mean(sigs)),
        })
        visited[idxs] = True
    return merged
