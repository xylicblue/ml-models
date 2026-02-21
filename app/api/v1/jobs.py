"""Job management API — submit jobs, poll status, download outputs."""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from app.jobs.models import JobRecord, JobStatus

router = APIRouter()

# These will be set by main.py during lifespan
_dispatcher = None
_temp_store = None


def set_dispatcher(dispatcher):
    global _dispatcher
    _dispatcher = dispatcher


def set_temp_store(store):
    global _temp_store
    _temp_store = store


class JobSubmitRequest(BaseModel):
    model_id: str
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    params: dict = {}


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str


@router.post("/jobs", response_model=JobSubmitResponse)
async def submit_job(request: JobSubmitRequest):
    """Submit a new analysis job."""
    if _dispatcher is None:
        raise HTTPException(status_code=503, detail="Job dispatcher not initialized")

    job = JobRecord(
        model_id=request.model_id,
        input_params={
            "image_path": request.image_path,
            "image_url": request.image_url,
            **request.params,
        },
    )
    job_id = await _dispatcher.submit(job)
    return JobSubmitResponse(
        job_id=job_id,
        status="pending",
        message="Job submitted successfully. Poll GET /api/v1/jobs/{id} for status.",
    )


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the current status and results of a job."""
    if _dispatcher is None:
        raise HTTPException(status_code=503, detail="Job dispatcher not initialized")

    job = await _dispatcher.get_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {
        "job_id": job.id,
        "model_id": job.model_id,
        "status": job.status.value,
        "progress": {
            "current": job.progress_current,
            "total": job.progress_total,
            "message": job.progress_message,
        },
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }

    if job.status == JobStatus.COMPLETED and job.result:
        # Include results but exclude large arrays (heatmap)
        result_summary = {
            k: v for k, v in job.result.items()
            if k not in ("heatmap", "points")
        }
        result_summary["point_count"] = len(job.result.get("points", []))
        response["result"] = result_summary
        response["output_files"] = job.output_files

    if job.status == JobStatus.FAILED:
        response["error"] = job.error

    return response


@router.get("/jobs/{job_id}/outputs/{filename}")
async def get_job_output(job_id: str, filename: str):
    """Download a result file (e.g., annotated image) from a completed job."""
    if _temp_store is None:
        raise HTTPException(status_code=503, detail="Result store not initialized")

    if not _temp_store.file_exists(job_id, filename):
        raise HTTPException(status_code=404, detail="Output file not found")

    path = _temp_store.get_output_path(job_id, filename)
    media_type = "image/png" if filename.endswith(".png") else "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=filename)
