"""AgriPay Compute Backend - FastAPI application."""

import os
import torch
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.config import settings
from app.api.v1.router import v1_router, upload_router_compat
from app.api.v1.health import router as health_root_router
from app.api.v1 import jobs as jobs_api
from app.api.v1 import analysis as analysis_api
from app.api.v1 import upload as upload_api
from app.models.registry import registry
from app.jobs.models import JobRecord, JobStatus
from app.jobs.in_process_queue import InProcessQueue
from app.storage.temp_results import temp_store
from app.processing.visualization import (
    render_counting_overlay,
    render_heatmap,
    render_size_annotated,
    render_size_color_coded,
)


def run_model_job(job: JobRecord) -> JobRecord:
    """Worker function: runs model inference for a job.

    Called by the InProcessQueue via run_in_executor (runs in a thread).
    This is intentionally synchronous — all operations are CPU/GPU bound.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Find and load the model
    model = registry.ensure_loaded(job.model_id, device=device)

    # Determine input
    image_path = job.input_params.get("image_path")
    image_url = job.input_params.get("image_url")
    input_data = image_path or image_url
    if not input_data:
        job.status = JobStatus.FAILED
        job.error = "No image_path or image_url provided"
        job.completed_at = datetime.utcnow()
        return job

    # Progress callback
    def on_progress(current, total, message):
        job.progress_current = current
        job.progress_total = total
        job.progress_message = message

    # Run inference
    result = model.predict(input_data, progress_cb=on_progress)

    # Save visualization outputs
    output_files = []
    if "heatmap" in result and "points" in result:
        job_dir = temp_store.get_job_dir(job.id)

        # Disable PIL decompression bomb check — drone imagery is trusted and large
        Image.MAX_IMAGE_PIXELS = None

        # Load the original uploaded image for overlays.
        # Cap at VIZ_MAX_DIM so we never render a 100M-pixel canvas.
        VIZ_MAX_DIM = 2048
        W = result.get("image_width", 1024)
        H = result.get("image_height", 1024)

        image_path = job.input_params.get("image_path")
        scale = 1.0
        if image_path and os.path.exists(image_path):
            try:
                base_img = Image.open(image_path).convert("RGB")
                orig_w, orig_h = base_img.size
                scale = min(1.0, VIZ_MAX_DIM / max(orig_w, orig_h))
                if scale < 1.0:
                    viz_w = max(1, int(orig_w * scale))
                    viz_h = max(1, int(orig_h * scale))
                    base_img = base_img.resize((viz_w, viz_h), Image.LANCZOS)
                    print(f"  Viz: downsampled {orig_w}x{orig_h} -> {viz_w}x{viz_h} (scale={scale:.3f})")
            except Exception as exc:
                print(f"  Viz: could not open image ({exc}), using blank canvas")
                scale = min(1.0, VIZ_MAX_DIM / max(W, H))
                viz_w = max(1, int(W * scale))
                viz_h = max(1, int(H * scale))
                base_img = Image.new("RGB", (viz_w, viz_h), (40, 40, 40))
        else:
            scale = min(1.0, VIZ_MAX_DIM / max(W, H))
            viz_w = max(1, int(W * scale))
            viz_h = max(1, int(H * scale))
            base_img = Image.new("RGB", (viz_w, viz_h), (40, 40, 40))

        # Scale point coordinates and sizes to match the visualization resolution
        points = [
            {"x": int(p["x"] * scale), "y": int(p["y"] * scale), "base_sigma": p.get("base_sigma", 0) * scale}
            for p in result["points"]
        ]
        plant_sizes = [s * scale for s in result.get("plant_sizes", [])]

        # Counting overlay
        overlay = render_counting_overlay(base_img, points)
        overlay_path = f"{job_dir}/counting_overlay.png"
        overlay.save(overlay_path)
        output_files.append("counting_overlay.png")

        # Heatmap
        heatmap_img = render_heatmap(result["heatmap"])
        heatmap_path = f"{job_dir}/density_heatmap.png"
        heatmap_img.save(heatmap_path)
        output_files.append("density_heatmap.png")

        if plant_sizes:
            # Size annotated
            annot = render_size_annotated(base_img, points, plant_sizes)
            annot_path = f"{job_dir}/size_annotated.png"
            annot.save(annot_path)
            output_files.append("size_annotated.png")

            # Color coded
            color = render_size_color_coded(base_img, points, plant_sizes)
            color_path = f"{job_dir}/size_color_coded.png"
            color.save(color_path)
            output_files.append("size_color_coded.png")

    # Store results (exclude large numpy arrays)
    result_serializable = {
        k: v for k, v in result.items()
        if k != "heatmap"
    }
    # Convert points to serializable format
    if "points" in result_serializable:
        result_serializable["points"] = [
            {"x": p["x"], "y": p["y"], "base_sigma": p["base_sigma"]}
            for p in result_serializable["points"]
        ]

    job.result = result_serializable
    job.output_files = output_files
    job.status = JobStatus.COMPLETED
    job.completed_at = datetime.utcnow()
    return job


# Global dispatcher reference
_dispatcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    global _dispatcher

    print(f"Starting AgriPay Compute Backend on port {settings.compute_port}")
    print(f"Dispatch mode: {settings.compute_dispatch_mode}")
    print(f"Imagery dir: {settings.imagery_dir}")
    print(f"Model weights dir: {settings.model_weights_dir}")

    # Discover models
    print("Discovering models...")
    registry.discover()
    print(f"  Found {len(registry.list_models())} model(s)")

    # Start job dispatcher
    _dispatcher = InProcessQueue(worker_fn=run_model_job)
    await _dispatcher.start()
    print("Job dispatcher started")

    # Wire dispatcher and temp store into API endpoints
    jobs_api.set_dispatcher(_dispatcher)
    jobs_api.set_temp_store(temp_store)
    analysis_api.set_dispatcher(_dispatcher)
    upload_api.set_dispatcher(_dispatcher)

    yield

    # Shutdown
    print("Shutting down AgriPay Compute Backend")
    await _dispatcher.stop()
    temp_store.cleanup_expired()


app = FastAPI(
    title="AgriPay Compute Service",
    description="Multi-source AI verification system for agricultural milestone validation",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server and any configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(health_root_router, tags=["health"])  # GET /health at root
app.include_router(v1_router)  # All /api/v1/* endpoints
app.include_router(upload_router_compat)  # /upload, /status, /download compat layer
