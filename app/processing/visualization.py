"""Visualization utilities for plant counting results."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import matplotlib.cm


def render_counting_overlay(
    base_image: Image.Image,
    points: List[Dict],
    dot_radius: int = 5,
    dot_color: Tuple[int, ...] = (255, 0, 0, 128),
) -> Image.Image:
    """Draw detected plant locations as dots on the image."""
    overlay = base_image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    for p in points:
        x, y = p["x"], p["y"]
        draw.ellipse(
            (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
            fill=dot_color,
        )
    # Add count text
    font = _get_font(32)
    count_text = f"Total Count: {len(points)}"
    draw.text((20, 20), count_text, fill=(255, 255, 255, 255), font=font)
    return overlay.convert("RGB")


def render_size_annotated(
    base_image: Image.Image,
    points: List[Dict],
    plant_sizes: List[float],
) -> Image.Image:
    """Draw plant bounding boxes with size annotations."""
    annot = base_image.copy().convert("RGBA")
    draw = ImageDraw.Draw(annot, "RGBA")
    font_small = _get_font(16)
    font_large = _get_font(32)

    for p, size in zip(points, plant_sizes):
        x, y = p["x"], p["y"]
        half = size / 2
        draw.rectangle(
            [x - half, y - half, x + half, y + half],
            outline=(255, 0, 0, 255),
            width=1,
        )
        draw.text(
            (x, y), f"{size:.1f}",
            fill=(255, 255, 0, 255), font=font_small, anchor="mm",
        )

    if plant_sizes:
        avg = np.mean(plant_sizes)
        draw.text((20, 20), f"Average Size: {avg:.2f}", fill=(255, 255, 255, 255), font=font_large)

    return annot.convert("RGB")


def render_size_color_coded(
    base_image: Image.Image,
    points: List[Dict],
    plant_sizes: List[float],
) -> Image.Image:
    """Draw plants color-coded by size: red=small, orange=medium, yellow=large."""
    color_img = base_image.copy().convert("RGBA")
    draw = ImageDraw.Draw(color_img, "RGBA")

    if len(plant_sizes) > 1:
        lower_q = float(np.percentile(plant_sizes, 25))
        upper_q = float(np.percentile(plant_sizes, 75))
    elif len(plant_sizes) == 1:
        lower_q = upper_q = plant_sizes[0]
    else:
        return color_img.convert("RGB")

    for p, size in zip(points, plant_sizes):
        x, y = p["x"], p["y"]
        if size < lower_q:
            color = (255, 0, 0, 255)
        elif size > upper_q:
            color = (255, 255, 0, 255)
        else:
            color = (255, 165, 0, 255)
        half = size / 2
        draw.rectangle([x - half, y - half, x + half, y + half], outline=color, width=1)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    return color_img.convert("RGB")


def render_heatmap(heatmap: np.ndarray) -> Image.Image:
    """Render a density heatmap as a jet-colored image."""
    h_min = np.nanmin(heatmap)
    h_max = np.nanmax(heatmap)
    denom = h_max - h_min
    if denom < 1e-8:
        denom = 1.0
    norm_heat = (heatmap - h_min) / denom
    cmap = matplotlib.colormaps["jet"]
    heat_rgba = (cmap(norm_heat) * 255).astype(np.uint8)
    return Image.fromarray(heat_rgba).convert("RGB")


def _get_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    """Try to load a TrueType font, falling back to default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return None
