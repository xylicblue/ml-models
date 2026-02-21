"""Models API — list registered models."""

from fastapi import APIRouter, Depends
from typing import Optional

from app.models.registry import registry

router = APIRouter()


@router.get("/models")
async def list_models(
    crop: Optional[str] = None,
    task: Optional[str] = None,
):
    """List all registered models with optional filtering."""
    specs = registry.list_models(crop=crop, task=task)
    return {
        "models": [
            {
                "model_id": s.model_id,
                "name": s.name,
                "crop": s.crop,
                "task": s.task,
                "input_type": s.input_type.value,
                "output_types": [o.value for o in s.output_types],
                "description": s.description,
                "version": s.version,
            }
            for s in specs
        ],
        "count": len(specs),
    }
