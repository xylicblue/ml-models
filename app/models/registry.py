"""Model registry with auto-discovery, lazy loading, and LRU eviction."""

import importlib
import inspect
import pkgutil
from collections import OrderedDict
from typing import Dict, List, Optional

from app.config import settings
from app.models.base import BaseModel, ModelSpec


class ModelRegistry:
    """Discovers, manages, and serves ML models.

    - Auto-discovers BaseModel subclasses in app/models/
    - Lazy-loads weights on first inference call
    - LRU eviction when max_loaded_models is exceeded
    """

    def __init__(self, max_loaded: int = 3):
        self._models: Dict[str, BaseModel] = {}
        self._loaded_order: OrderedDict[str, None] = OrderedDict()
        self._max_loaded = max_loaded

    def discover(self) -> None:
        """Scan app.models package for BaseModel subclasses and register them."""
        import app.models as models_pkg

        for importer, modname, ispkg in pkgutil.walk_packages(
            models_pkg.__path__, prefix="app.models."
        ):
            if ispkg:
                continue
            # Skip base.py, registry.py, and nn/ subpackage internals
            if modname in ("app.models.base", "app.models.registry"):
                continue
            if ".nn." in modname:
                continue
            try:
                mod = importlib.import_module(modname)
            except Exception as e:
                print(f"Warning: failed to import {modname}: {e}")
                continue

            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if (
                    issubclass(obj, BaseModel)
                    and obj is not BaseModel
                    and not inspect.isabstract(obj)
                ):
                    instance = obj()
                    model_id = instance.spec().model_id
                    self._models[model_id] = instance
                    print(f"  Registered model: {model_id} ({instance.spec().name})")

    def list_models(
        self,
        crop: Optional[str] = None,
        task: Optional[str] = None,
    ) -> List[ModelSpec]:
        """List registered models, optionally filtered by crop and/or task."""
        specs = [m.spec() for m in self._models.values()]
        if crop:
            specs = [s for s in specs if s.crop == crop]
        if task:
            specs = [s for s in specs if s.task == task]
        return specs

    def get(self, model_id: str) -> Optional[BaseModel]:
        """Get a model by ID."""
        return self._models.get(model_id)

    def ensure_loaded(self, model_id: str, device: str = "cpu") -> BaseModel:
        """Ensure a model is loaded, loading it if necessary.

        Uses LRU eviction to stay within max_loaded_models limit.
        """
        model = self._models.get(model_id)
        if model is None:
            raise ValueError(f"Model '{model_id}' not found in registry")

        if model.is_loaded():
            # Move to end of LRU
            if model_id in self._loaded_order:
                self._loaded_order.move_to_end(model_id)
            return model

        # Evict oldest if at capacity
        while len(self._loaded_order) >= self._max_loaded:
            oldest_id, _ = self._loaded_order.popitem(last=False)
            oldest_model = self._models.get(oldest_id)
            if oldest_model and oldest_model.is_loaded():
                print(f"  Evicting model: {oldest_id}")
                oldest_model.unload()

        # Load the model
        print(f"  Loading model: {model_id} on {device}")
        model.load(device)
        self._loaded_order[model_id] = None
        return model

    def loaded_count(self) -> int:
        return len(self._loaded_order)

    def loaded_models(self) -> List[str]:
        return list(self._loaded_order.keys())


# Global registry instance
registry = ModelRegistry(max_loaded=settings.max_loaded_models)
