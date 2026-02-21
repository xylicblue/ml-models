"""Health check endpoint."""

from fastapi import APIRouter
import torch
import platform
import sys

router = APIRouter()


@router.get("/health")
async def health_check():
    """Service health, GPU status, and system info."""
    gpu_available = torch.cuda.is_available()
    gpu_info = None
    if gpu_available:
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_total_mb": round(torch.cuda.get_device_properties(0).total_mem / 1024 / 1024),
            "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024),
            "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1024 / 1024),
        }

    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_info": gpu_info,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if gpu_available else None,
        "python_version": sys.version,
        "platform": platform.platform(),
    }
