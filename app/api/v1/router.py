"""Aggregate all v1 API routers."""

from fastapi import APIRouter
from app.api.v1.health import router as health_router
from app.api.v1.models_api import router as models_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.analysis import router as analysis_router
from app.api.v1.verification import router as verification_router
from app.api.v1.upload import router as upload_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(health_router, tags=["health"])
v1_router.include_router(models_router, tags=["models"])
v1_router.include_router(jobs_router, tags=["jobs"])
v1_router.include_router(analysis_router, tags=["analysis"])
v1_router.include_router(verification_router, tags=["verification"])

# Compatibility shim — mounts /upload, /status/{id}, /download/{id}/{type} at root
upload_router_compat = APIRouter()
upload_router_compat.include_router(upload_router, tags=["upload"])
