"""Wheat plant counting and size estimation model.

Wraps the VGG16-BN encoder-decoder for use in the model registry.
Handles the full pipeline: tiling, inference, peak detection, NMS, size estimation.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from scipy.spatial import KDTree
from torchvision.transforms import functional as TF

from app.config import settings
from app.io.image_reader import GeoTiffReader
from app.models.base import (
    BaseModel,
    InputType,
    ModelSpec,
    OutputType,
    ProgressCallback,
)
from app.models.nn.vgg_encoder_decoder import Model
from app.processing.peak_detection import get_peaks, nms_merge_points
from app.processing.sliding_window import cosine_weight_2d, sliding_windows

# Processing parameters
TILE_SIZE = 1024
OVERLAP = 128
PEAK_THRESHOLD = 0.004
BORDER_MARGIN = 10
GLOBAL_PADDING = 10
MERGE_RADIUS = 30
COUNT_MODEL_SIZE = (320, 320)  # (w, h)
SIZE_MODEL_SIZE = (400, 400)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class WheatPlantCounter(BaseModel):
    """Plant counting + size estimation for wheat crops using VGG16-BN."""

    def __init__(self):
        self._count_model: Optional[Model] = None
        self._size_model: Optional[Model] = None

    def spec(self) -> ModelSpec:
        return ModelSpec(
            model_id="wheat_plant_counter_v1",
            name="Wheat Plant Counter",
            crop="wheat",
            task="plant_counting",
            input_type=InputType.GEOTIFF,
            output_types=[
                OutputType.PLANT_COUNT,
                OutputType.DENSITY_MAP,
                OutputType.SIZE_ESTIMATION,
            ],
            weight_files=[
                "checkpoint_counting.pth",
                "checkpoint_size_estimation.pth",
            ],
            description="Detects and counts individual wheat plants from drone imagery. "
                        "Also estimates plant size using a separate regression model.",
            version="1.0.0",
            input_size=COUNT_MODEL_SIZE,
        )

    def load(self, device: str = "cpu") -> None:
        weights_dir = settings.model_weights_dir
        count_path = os.path.join(weights_dir, "checkpoint_counting.pth")
        size_path = os.path.join(weights_dir, "checkpoint_size_estimation.pth")

        if not os.path.exists(count_path):
            raise FileNotFoundError(f"Counting weights not found: {count_path}")
        if not os.path.exists(size_path):
            raise FileNotFoundError(f"Size weights not found: {size_path}")

        self._device = device

        # Skip VGG pretrained download — checkpoint already contains all weights
        self._count_model = Model(gap=False, load_pretrained_vgg=False).to(device)
        self._count_model.load_state_dict(
            torch.load(count_path, map_location=device, weights_only=True)
        )
        self._count_model.eval()

        self._size_model = Model(gap=True, load_pretrained_vgg=False).to(device)
        self._size_model.load_state_dict(
            torch.load(size_path, map_location=device, weights_only=True)
        )
        self._size_model.eval()

        self._loaded = True

    def unload(self) -> None:
        self._count_model = None
        self._size_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False

    def predict(
        self,
        input_data: Any,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Run plant counting on a GeoTIFF or PIL image.

        input_data: str (file path) or PIL.Image
        Returns dict with plant_count, points, plant_sizes, heatmap, etc.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Open image
        if isinstance(input_data, str):
            reader = GeoTiffReader(input_data, band_preset="auto")
            W, H = reader.width, reader.height
            use_reader = True
        elif isinstance(input_data, Image.Image):
            input_data = input_data.convert("RGB")
            W, H = input_data.size
            use_reader = False
            reader = None
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        # Prepare accumulators
        heat_sum = np.zeros((H, W), dtype=np.float32)
        heat_wsum = np.zeros((H, W), dtype=np.float32)
        global_points: List[Dict] = []
        base_weight = cosine_weight_2d(TILE_SIZE, TILE_SIZE)

        # Count total tiles for progress
        tiles = list(sliding_windows(W, H, TILE_SIZE, OVERLAP))
        total_tiles = len(tiles)

        try:
            for idx, (x0, y0, tw, th) in enumerate(tiles):
                if progress_cb:
                    progress_cb(idx + 1, total_tiles, f"Processing tile {idx+1}/{total_tiles}")

                # Read tile
                if use_reader:
                    tile_rgb = reader.read_tile(x0, y0, tw, th, normalize_uint8=True)
                    nodata_mask = reader.get_nodata_mask(x0, y0, tw, th)
                    tile_img = Image.fromarray(tile_rgb)
                else:
                    tile_img = input_data.crop((x0, y0, x0 + tw, y0 + th))
                    nodata_mask = None

                # Run counting model
                tile_resized = tile_img.resize(COUNT_MODEL_SIZE, Image.BILINEAR)
                t = TF.to_tensor(tile_resized)
                t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
                t = t.unsqueeze(0).to(self._device)

                # Run size model
                tile_resized_size = tile_img.resize(SIZE_MODEL_SIZE, Image.BILINEAR)
                ts = TF.to_tensor(tile_resized_size)
                ts = TF.normalize(ts, IMAGENET_MEAN, IMAGENET_STD)
                ts = ts.unsqueeze(0).to(self._device)

                with torch.no_grad():
                    output, _ = self._count_model(t)
                    density_map = np.ascontiguousarray(output.squeeze().cpu().numpy(), dtype=np.float32)
                    pre_dis = self._size_model(ts)
                    pre_dis = float(pre_dis.item())

                # Scale size estimate to tile coordinates
                scale_factor_size = ((tw / SIZE_MODEL_SIZE[0]) + (th / SIZE_MODEL_SIZE[1])) / 2.0
                tile_sigma = pre_dis * scale_factor_size

                # Find peaks in density map
                points_dm = get_peaks(density_map, threshold=PEAK_THRESHOLD)
                dm_h, dm_w = density_map.shape

                scale_x = tw / dm_w
                scale_y = th / dm_h
                mapped_points = [(int(x * scale_x), int(y * scale_y)) for (y, x) in points_dm]

                # Filter border points
                tile_points = [
                    (x, y) for (x, y) in mapped_points
                    if BORDER_MARGIN <= x < tw - BORDER_MARGIN
                    and BORDER_MARGIN <= y < th - BORDER_MARGIN
                ]

                # Filter global padding
                tile_points = [
                    (x, y) for (x, y) in tile_points
                    if (GLOBAL_PADDING <= (x0 + x) < (W - GLOBAL_PADDING))
                    and (GLOBAL_PADDING <= (y0 + y) < (H - GLOBAL_PADDING))
                ]

                # Filter nodata regions
                if nodata_mask is not None:
                    tile_points = [
                        (x, y) for (x, y) in tile_points
                        if not nodata_mask[min(y, th - 1), min(x, tw - 1)]
                    ]

                for (x, y) in tile_points:
                    global_points.append({
                        "x": x0 + x,
                        "y": y0 + y,
                        "base_sigma": tile_sigma,
                    })

                # Accumulate density heatmap with cosine blending
                dm_img = Image.fromarray(density_map.astype(np.float32))
                dm_img = dm_img.resize((tw, th), Image.BILINEAR)
                dm = np.array(dm_img, dtype=np.float32)

                w2d = base_weight[:th, :tw]

                # Mask nodata from weight accumulation
                if nodata_mask is not None:
                    w2d = w2d.copy()
                    w2d[nodata_mask] = 0

                heat_sum[y0:y0 + th, x0:x0 + tw] += dm * w2d
                heat_wsum[y0:y0 + th, x0:x0 + tw] += w2d
        finally:
            # Ensure reader is always closed, even on exception
            if use_reader and reader:
                reader.close()

        # Normalize heatmap
        mask = heat_wsum > 1e-6
        heatmap = np.zeros_like(heat_sum, dtype=np.float32)
        heatmap[mask] = heat_sum[mask] / heat_wsum[mask]

        # Merge nearby detections
        merged_points = nms_merge_points(global_points, radius=MERGE_RADIUS)

        # Compute adaptive plant sizes
        plant_sizes = self._compute_sizes(merged_points)

        # Statistics
        plant_count = len(merged_points)
        avg_size = float(np.mean(plant_sizes)) if plant_sizes else 0.0
        min_size = float(min(plant_sizes)) if plant_sizes else 0.0
        max_size = float(max(plant_sizes)) if plant_sizes else 0.0

        return {
            "plant_count": plant_count,
            "points": merged_points,
            "plant_sizes": plant_sizes,
            "heatmap": heatmap,
            "avg_size_px": avg_size,
            "min_size_px": min_size,
            "max_size_px": max_size,
            "image_width": W,
            "image_height": H,
            "tiles_processed": total_tiles,
        }

    def _compute_sizes(self, points: List[Dict]) -> List[float]:
        """Compute adaptive plant sizes using KDTree nearest-neighbor blending."""
        if not points:
            return []

        coords = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
        tree = KDTree(coords)
        num_points = len(coords)
        k = min(num_points, 3)

        if k > 1:
            distances, _ = tree.query(coords, k=k)
            local_mean = 0.8 * np.mean(distances[:, 1:k], axis=1) * 2
        else:
            local_mean = np.full(num_points, np.inf, dtype=np.float32)

        plant_sizes = []
        for i, p in enumerate(points):
            dm = local_mean[i]
            base = p["base_sigma"]
            dis = 0.5 * base + 0.5 * (dm if np.isfinite(dm) else base)
            plant_sizes.append(float(dis))

        return plant_sizes
