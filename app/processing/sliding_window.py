"""Sliding window utilities for processing large images tile by tile."""

from typing import Iterator, Tuple
import numpy as np


def sliding_windows(
    width: int,
    height: int,
    tile_size: int = 1024,
    overlap: int = 128,
) -> Iterator[Tuple[int, int, int, int]]:
    """Generate (x0, y0, tile_w, tile_h) for sliding window traversal.

    Ensures full coverage — adds an extra tile at the right/bottom edge
    if the last regular tile doesn't reach the boundary.
    """
    step = tile_size - overlap
    xs = list(range(0, max(1, width - tile_size + 1), step))
    ys = list(range(0, max(1, height - tile_size + 1), step))
    if xs[-1] + tile_size < width:
        xs.append(width - tile_size)
    if ys[-1] + tile_size < height:
        ys.append(height - tile_size)
    for y0 in ys:
        for x0 in xs:
            tw = min(tile_size, width - x0)
            th = min(tile_size, height - y0)
            yield x0, y0, tw, th


def cosine_weight_2d(height: int, width: int) -> np.ndarray:
    """Create a 2D cosine blending weight matrix.

    Values are 0 at edges and 1 at center, used for smooth tile stitching.
    """
    y = np.linspace(0, np.pi, height, dtype=np.float32)
    x = np.linspace(0, np.pi, width, dtype=np.float32)
    wy = (1 - np.cos(y)) / 2.0
    wx = (1 - np.cos(x)) / 2.0
    w2d = np.outer(wy, wx)
    max_val = w2d.max()
    if max_val > 0:
        w2d /= max_val
    return w2d
