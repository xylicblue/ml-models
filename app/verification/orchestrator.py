"""Milestone verification orchestrator.

Coordinates multi-source analysis (satellite + drone + IoT) and
writes the composite verdict to the database.
"""

from datetime import datetime
from typing import Any, Dict

from app.db.supabase_client import get_supabase
from app.verification.satellite_analyzer import analyze_satellite
from app.verification.drone_analyzer import analyze_drone
from app.verification.iot_analyzer import analyze_iot
from app.verification.scorer import compute_verdict


async def verify_milestone(milestone_id: str) -> Dict[str, Any]:
    """Run full multi-source verification for a milestone.

    Steps:
    1. Fetch milestone context from DB
    2. Run satellite, drone, IoT analysis in parallel
    3. Compute weighted verdict
    4. Write results to DB (agro_augmentation + verifications)
    5. Return verdict

    Returns:
        The complete verification result dict
    """
    supabase = get_supabase()

    # 1. Fetch milestone details
    milestone_response = (
        supabase.table("cycle_milestones")
        .select(
            "id, crop_cycle_id, template_id, status, "
            "crop_cycles(id, farm_id, crop_id, crops(name)), "
            "milestone_templates(name, expected_ndvi_change, verification_type)"
        )
        .eq("id", milestone_id)
        .single()
        .execute()
    )

    if not milestone_response.data:
        return {
            "verdict": "ERROR",
            "error": f"Milestone {milestone_id} not found",
        }

    milestone = milestone_response.data
    cycle = milestone.get("crop_cycles", {})
    farm_id = cycle.get("farm_id")
    template = milestone.get("milestone_templates", {})
    milestone_type = template.get("name", "unknown").lower().replace(" ", "_")
    expected_ndvi = template.get("expected_ndvi_change")

    if not farm_id:
        return {
            "verdict": "ERROR",
            "error": "Could not determine farm_id for this milestone",
        }

    # 2. Run all analyzers
    satellite_result = await analyze_satellite(
        farm_id=farm_id,
        milestone_type=milestone_type,
        expected_ndvi_change=expected_ndvi,
    )

    drone_result = await analyze_drone(
        farm_id=farm_id,
        milestone_id=milestone_id,
        milestone_type=milestone_type,
    )

    iot_result = await analyze_iot(
        farm_id=farm_id,
        milestone_type=milestone_type,
    )

    # 3. Compute verdict
    verdict_result = compute_verdict(satellite_result, drone_result, iot_result)
    verdict_result["processed_at"] = datetime.utcnow().isoformat()

    # 4. Write to database
    try:
        # Update milestone with agro_augmentation
        supabase.table("cycle_milestones").update({
            "agro_augmentation": verdict_result,
            "status": "pending_verification",
            "updated_at": datetime.utcnow().isoformat(),
        }).eq("id", milestone_id).execute()

        # Create verification record
        supabase.table("verifications").insert({
            "milestone_id": milestone_id,
            "source": "system",
            "evidence_data": verdict_result,
            "ai_confidence": verdict_result["overall_confidence"],
            "ai_analysis": {
                "satellite": satellite_result,
                "drone": drone_result,
                "iot": iot_result,
            },
            "status": _verdict_to_verification_status(verdict_result["verdict"]),
        }).execute()

        # Create evidence records for each source that had data
        for source_name, source_result in [
            ("satellite", satellite_result),
            ("drone_output", drone_result),
            ("iot_reading", iot_result),
        ]:
            if source_result.get("status") == "ANALYZED":
                # Get the verification ID we just created
                latest_verification = (
                    supabase.table("verifications")
                    .select("id")
                    .eq("milestone_id", milestone_id)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )
                if latest_verification.data:
                    supabase.table("verification_evidence").insert({
                        "verification_id": latest_verification.data[0]["id"],
                        "evidence_type": source_name,
                        "evidence_source_id": milestone_id,
                        "captured_at": datetime.utcnow().isoformat(),
                        "relevance_score": source_result.get("confidence", 0),
                        "notes": source_result.get("interpretation", ""),
                    }).execute()

    except Exception as e:
        verdict_result["db_write_error"] = str(e)

    return verdict_result


def _verdict_to_verification_status(verdict: str) -> str:
    """Map verdict to verification status."""
    mapping = {
        "MILESTONE_COMPLETE": "approved",
        "MANUAL_REVIEW_REQUIRED": "manual_review_required",
        "MILESTONE_FAILED": "rejected",
        "INSUFFICIENT_EVIDENCE": "manual_review_required",
    }
    return mapping.get(verdict, "processing")
