"""GeoTIFF reader with rasterio windowed reads for memory-efficient processing.

Handles large drone imagery (200MB-GBs) by reading only the needed tile
from disk instead of loading the entire image into memory.
"""

from typing import Optional, Tuple, Dict
import numpy as np
import rasterio
from rasterio.windows import Window

# MicaSense RedEdge-MX Dual Camera System (10-band) band presets
# Maps logical RGB to MicaSense bands (1-indexed)
BAND_PRESETS = {
    "micasense_rgb": {
        "bands": [6, 4, 2],  # Red-668nm, Green-560nm, Blue-475nm
        "description": "MicaSense 10-band → visual RGB",
    },
    "micasense_cir": {
        "bands": [10, 6, 4],  # NIR-842nm, Red-668nm, Green-560nm
        "description": "MicaSense 10-band → Color Infrared",
    },
    "rgb": {
        "bands": [1, 2, 3],  # Standard 3-band RGB
        "description": "Standard 3-band RGB image",
    },
}


class GeoTiffReader:
    """Memory-efficient GeoTIFF reader using rasterio windowed reads.

    Usage:
        reader = GeoTiffReader("path/to/image.tif", band_preset="micasense_rgb")

        # Read a single tile
        tile = reader.read_tile(x0=0, y0=0, width=1024, height=1024)

        # Get image dimensions
        w, h = reader.width, reader.height
    """

    def __init__(
        self,
        path: str,
        band_preset: str = "auto",
        nodata_value: Optional[float] = None,
    ):
        self.path = path
        self._src: Optional[rasterio.DatasetReader] = None

        # Open the file
        self._src = rasterio.open(path)

        # Determine band selection
        if band_preset == "auto":
            band_preset = self._detect_preset()
        preset = BAND_PRESETS.get(band_preset)
        if preset is None:
            raise ValueError(
                f"Unknown band preset '{band_preset}'. "
                f"Available: {list(BAND_PRESETS.keys())}"
            )
        self.bands = preset["bands"]
        self.band_preset = band_preset

        # Nodata value
        if nodata_value is not None:
            self.nodata = nodata_value
        elif self._src.nodata is not None:
            self.nodata = self._src.nodata
        else:
            self.nodata = None

        # Compute percentile stretch values for normalization (lazy)
        self._stretch_min: Optional[np.ndarray] = None
        self._stretch_max: Optional[np.ndarray] = None

    def _detect_preset(self) -> str:
        """Auto-detect band preset from file metadata."""
        if self._src.count >= 10:
            return "micasense_rgb"
        elif self._src.count >= 3:
            return "rgb"
        else:
            return "rgb"

    @property
    def width(self) -> int:
        return self._src.width

    @property
    def height(self) -> int:
        return self._src.height

    @property
    def count(self) -> int:
        return self._src.count

    @property
    def dtype(self) -> str:
        return self._src.dtypes[0]

    @property
    def crs(self) -> Optional[str]:
        return str(self._src.crs) if self._src.crs else None

    @property
    def bounds(self) -> Dict:
        b = self._src.bounds
        return {"left": b.left, "bottom": b.bottom, "right": b.right, "top": b.top}

    def read_tile(
        self,
        x0: int,
        y0: int,
        width: int,
        height: int,
        normalize_uint8: bool = True,
    ) -> np.ndarray:
        """Read a tile as (H, W, 3) uint8 RGB array.

        Args:
            x0, y0: Top-left pixel coordinate
            width, height: Tile dimensions in pixels
            normalize_uint8: If True, convert to uint8 with percentile stretch
        """
        window = Window(col_off=x0, row_off=y0, width=width, height=height)
        # rasterio reads as (bands, H, W)
        tile = self._src.read(self.bands, window=window)

        # Handle nodata
        if self.nodata is not None:
            nodata_mask = np.any(tile == self.nodata, axis=0)
            tile[:, nodata_mask] = 0

        if normalize_uint8 and tile.dtype != np.uint8:
            tile = self._normalize_to_uint8(tile)

        # Transpose to (H, W, C)
        return np.transpose(tile, (1, 2, 0))

    def read_tile_raw(
        self,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Read a raw tile as (bands, H, W) in original dtype. No normalization."""
        window = Window(col_off=x0, row_off=y0, width=width, height=height)
        return self._src.read(self.bands, window=window)

    def get_nodata_mask(
        self,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Get a boolean mask where True = nodata pixel for a given tile."""
        window = Window(col_off=x0, row_off=y0, width=width, height=height)
        tile = self._src.read(self.bands, window=window)
        if self.nodata is not None:
            return np.any(tile == self.nodata, axis=0)
        return np.zeros((height, width), dtype=bool)

    def _compute_stretch(self):
        """Compute 2nd-98th percentile for normalization from a downsampled read."""
        # Read a downsampled version to avoid loading the whole image
        factor = max(1, max(self.width, self.height) // 2048)
        out_w = self.width // factor
        out_h = self.height // factor
        data = self._src.read(
            self.bands,
            out_shape=(len(self.bands), out_h, out_w),
        )
        if self.nodata is not None:
            valid_mask = ~np.any(data == self.nodata, axis=0)
        else:
            valid_mask = np.ones((out_h, out_w), dtype=bool)

        mins = []
        maxs = []
        for band_idx in range(len(self.bands)):
            band_data = data[band_idx][valid_mask]
            if len(band_data) > 0:
                mins.append(np.percentile(band_data, 2))
                maxs.append(np.percentile(band_data, 98))
            else:
                mins.append(0)
                maxs.append(1)

        self._stretch_min = np.array(mins, dtype=np.float32)
        self._stretch_max = np.array(maxs, dtype=np.float32)

    def _normalize_to_uint8(self, tile: np.ndarray) -> np.ndarray:
        """Normalize tile from original dtype to uint8 using percentile stretch.

        tile: (C, H, W) array
        """
        if self._stretch_min is None:
            self._compute_stretch()

        result = np.zeros_like(tile, dtype=np.float32)
        for i in range(tile.shape[0]):
            lo = self._stretch_min[i]
            hi = self._stretch_max[i]
            if hi - lo < 1e-6:
                result[i] = 0
            else:
                result[i] = (tile[i].astype(np.float32) - lo) / (hi - lo)

        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result

    def close(self):
        if self._src is not None:
            self._src.close()
            self._src = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
