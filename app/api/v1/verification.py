"""Milestone verification endpoint — orchestrates multi-source verification."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from app.verification.orchestrator import verify_milestone as run_verification

router = APIRouter()


class VerifyMilestoneRequest(BaseModel):
    milestone_id: str


@router.post("/verify-milestone")
async def verify_milestone(request: VerifyMilestoneRequest):
    """Trigger multi-source verification for a milestone.

    Orchestrates satellite + drone + IoT analysis and writes
    the composite verdict to the database.

    Returns immediately with the verification result.
    For async processing, use the /jobs endpoint instead.
    """
    try:
        result = await run_verification(request.milestone_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}",
        )

    if result.get("verdict") == "ERROR":
        raise HTTPException(
            status_code=404,
            detail=result.get("error", "Unknown error"),
        )

    return {
        "status": "verified",
        "verdict": result.get("verdict"),
        "overall_confidence": result.get("overall_confidence"),
        "recommendation": result.get("recommendation"),
        "report": result,
    }
