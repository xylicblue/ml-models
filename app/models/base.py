"""Base model interface and data types for the model registry."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class InputType(str, Enum):
    GEOTIFF = "geotiff"
    IMAGE_RGB = "image_rgb"
    TABULAR = "tabular"


class OutputType(str, Enum):
    PLANT_COUNT = "plant_count"
    DENSITY_MAP = "density_map"
    SIZE_ESTIMATION = "size_estimation"
    CLASSIFICATION = "classification"
    SCALAR = "scalar"


@dataclass
class ModelSpec:
    """Metadata describing a registered model."""
    model_id: str
    name: str
    crop: str
    task: str
    input_type: InputType
    output_types: List[OutputType]
    weight_files: List[str]
    description: str = ""
    version: str = "1.0.0"
    input_size: tuple = (320, 320)
    extra: Dict[str, Any] = field(default_factory=dict)


# Type alias for progress callbacks: fn(current_step, total_steps, message)
ProgressCallback = Callable[[int, int, str], None]


class BaseModel(ABC):
    """Abstract base class for all ML models in the registry.

    To register a new model:
    1. Create a new .py file in app/models/
    2. Subclass BaseModel
    3. Implement spec(), load(), predict()
    4. The registry auto-discovers it at startup
    """

    _loaded: bool = False
    _device: str = "cpu"

    @abstractmethod
    def spec(self) -> ModelSpec:
        """Return model metadata."""
        ...

    @abstractmethod
    def load(self, device: str = "cpu") -> None:
        """Load model weights onto the given device."""
        ...

    @abstractmethod
    def predict(
        self,
        input_data: Any,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Run inference. Returns a dict of results."""
        ...

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        """Release model weights from memory."""
        self._loaded = False

    def get_device(self) -> str:
        return self._device
