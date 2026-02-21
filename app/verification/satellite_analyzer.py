"""Satellite data analyzer — rule-based NDVI/SAVI change detection.

Queries the `field_metrics` table for recent vegetation index data
and compares against milestone expectations.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from app.db.supabase_client import get_supabase


async def analyze_satellite(
    farm_id: str,
    milestone_type: str,
    expected_ndvi_change: Optional[float] = None,
    lookback_days: int = 14,
) -> Dict[str, Any]:
    """Analyze satellite data for milestone verification.

    Queries field_metrics for NDVI trends over the lookback period.
    Compares baseline vs current to determine if expected change occurred.

    Returns:
        dict with status, confidence, interpretation, and raw data
    """
    try:
        supabase = get_supabase()
    except RuntimeError:
        return {
            "status": "NO_DATA",
            "confidence": 0.0,
            "interpretation": "Database not configured",
            "data_points": 0,
        }

    cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()

    # Query field_metrics for the farm
    response = (
        supabase.table("field_metrics")
        .select("date, ndvi_mean, savi_mean, lai_mean, moisture_mean, cloud_coverage_pct")
        .eq("farm_id", farm_id)
        .gte("date", cutoff)
        .order("date")
        .execute()
    )

    records = response.data if response.data else []

    if len(records) < 2:
        return {
            "status": "NO_DATA",
            "confidence": 0.0,
            "interpretation": f"Insufficient satellite data: only {len(records)} data points in last {lookback_days} days",
            "data_points": len(records),
        }

    # Filter out records with high cloud coverage
    clean_records = [r for r in records if (r.get("cloud_coverage_pct") or 100) <= 30]
    if len(clean_records) < 2:
        clean_records = records  # Fall back to all records if too few clean ones

    # Compute baseline (first reading) and current (latest reading)
    baseline_ndvi = clean_records[0].get("ndvi_mean")
    current_ndvi = clean_records[-1].get("ndvi_mean")

    if baseline_ndvi is None or current_ndvi is None:
        return {
            "status": "NO_DATA",
            "confidence": 0.0,
            "interpretation": "NDVI values are missing from field_metrics records",
            "data_points": len(clean_records),
        }

    ndvi_change = current_ndvi - baseline_ndvi

    # Determine expected threshold
    if expected_ndvi_change is None:
        # Default expectations by milestone type
        expected_map = {
            "planting": 0.10,
            "fertilizer": 0.15,
            "irrigation": 0.05,
            "growth": 0.10,
            "harvest": -0.20,  # NDVI drops at harvest
        }
        expected_ndvi_change = expected_map.get(milestone_type, 0.10)

    # Score confidence based on how well the change matches expectations
    if milestone_type == "harvest":
        # Harvest expects a drop
        if ndvi_change <= expected_ndvi_change:
            confidence = min(1.0, abs(ndvi_change) / abs(expected_ndvi_change))
        else:
            confidence = max(0.0, 1.0 - (ndvi_change - expected_ndvi_change) / 0.2)
    else:
        # Other milestones expect an increase
        if ndvi_change >= expected_ndvi_change:
            confidence = min(1.0, ndvi_change / expected_ndvi_change)
        else:
            confidence = max(0.0, ndvi_change / expected_ndvi_change)

    # Interpret
    if confidence >= 0.8:
        interpretation = f"Satellite data strongly supports milestone completion. NDVI changed by {ndvi_change:+.3f} (expected {expected_ndvi_change:+.3f})"
    elif confidence >= 0.5:
        interpretation = f"Moderate satellite evidence. NDVI changed by {ndvi_change:+.3f} (expected {expected_ndvi_change:+.3f})"
    elif confidence > 0:
        interpretation = f"Weak satellite evidence. NDVI changed by {ndvi_change:+.3f} (expected {expected_ndvi_change:+.3f})"
    else:
        interpretation = f"No satellite evidence of milestone completion. NDVI changed by {ndvi_change:+.3f}"

    return {
        "status": "ANALYZED",
        "confidence": round(confidence, 3),
        "interpretation": interpretation,
        "ndvi_baseline": round(baseline_ndvi, 4),
        "ndvi_current": round(current_ndvi, 4),
        "ndvi_change": round(ndvi_change, 4),
        "expected_change": expected_ndvi_change,
        "data_points": len(clean_records),
        "date_range": {
            "start": clean_records[0]["date"],
            "end": clean_records[-1]["date"],
        },
    }
