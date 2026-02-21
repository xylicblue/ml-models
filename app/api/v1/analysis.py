"""Plant counting convenience endpoint."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.jobs.models import JobRecord

router = APIRouter()

# Set by main.py during lifespan (same pattern as jobs.py)
_dispatcher = None


def set_dispatcher(dispatcher):
    global _dispatcher
    _dispatcher = dispatcher


class PlantCountRequest(BaseModel):
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    model_id: str = "wheat_plant_counter_v1"


class PlantCountResponse(BaseModel):
    job_id: str
    status: str
    message: str


@router.post("/analyze/plant-count", response_model=PlantCountResponse)
async def analyze_plant_count(request: PlantCountRequest):
    """Submit a plant counting analysis job.

    Convenience wrapper around the generic /jobs endpoint
    that pre-selects the plant counting model.
    """
    if _dispatcher is None:
        raise HTTPException(status_code=503, detail="Job dispatcher not initialized")

    if not request.image_path and not request.image_url:
        raise HTTPException(
            status_code=400,
            detail="Either image_path or image_url must be provided",
        )

    job = JobRecord(
        model_id=request.model_id,
        input_params={
            "image_path": request.image_path,
            "image_url": request.image_url,
        },
    )
    job_id = await _dispatcher.submit(job)
    return PlantCountResponse(
        job_id=job_id,
        status="pending",
        message="Plant counting job submitted. Poll GET /api/v1/jobs/{id} for status.",
    )
