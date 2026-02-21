"""IoT sensor data analyzer — rule-based spike and trend detection.

Queries `iot_readings` for sensor data and performs statistical
analysis to detect patterns relevant to milestone verification.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np

from app.db.supabase_client import get_supabase


async def analyze_iot(
    farm_id: str,
    milestone_type: str,
    lookback_days: int = 14,
) -> Dict[str, Any]:
    """Analyze IoT sensor data for milestone verification.

    Queries iot_readings for the farm's devices, computes statistics,
    and detects anomalies/spikes relevant to the milestone type.
    """
    try:
        supabase = get_supabase()
    except RuntimeError:
        return {
            "status": "NO_DATA",
            "confidence": 0.0,
            "interpretation": "Database not configured",
            "readings_count": 0,
        }

    # Find active IoT devices for this farm
    devices_response = (
        supabase.table("iot_devices")
        .select("id, sensor_type_id, display_name, config")
        .eq("farm_id", farm_id)
        .eq("is_active", True)
        .execute()
    )

    devices = devices_response.data if devices_response.data else []
    if not devices:
        return {
            "status": "NO_DATA",
            "confidence": 0.0,
            "interpretation": "No active IoT devices found for this farm",
            "readings_count": 0,
        }

    # Query readings for all devices
    cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()
    device_ids = [d["id"] for d in devices]

    all_readings: List[Dict] = []
    for device_id in device_ids:
        response = (
            supabase.table("iot_readings")
            .select("device_id, captured_at, value, quality_flag")
            .eq("device_id", device_id)
            .gte("captured_at", cutoff)
            .eq("quality_flag", "normal")
            .order("captured_at")
            .execute()
        )
        if response.data:
            all_readings.extend(response.data)

    if not all_readings:
        return {
            "status": "NO_DATA",
            "confidence": 0.0,
            "interpretation": f"No IoT readings in the last {lookback_days} days",
            "readings_count": 0,
        }

    # Analyze readings
    values = [r["value"] for r in all_readings]
    values_arr = np.array(values, dtype=np.float64)

    mean_val = float(np.mean(values_arr))
    std_val = float(np.std(values_arr))
    min_val = float(np.min(values_arr))
    max_val = float(np.max(values_arr))

    # Split into first half (baseline) and second half (recent)
    mid = len(values_arr) // 2
    if mid > 0:
        baseline_mean = float(np.mean(values_arr[:mid]))
        recent_mean = float(np.mean(values_arr[mid:]))
        change_pct = ((recent_mean - baseline_mean) / (baseline_mean + 1e-6)) * 100
    else:
        baseline_mean = mean_val
        recent_mean = mean_val
        change_pct = 0.0

    # Detect spikes (values > baseline + 2*std)
    threshold = baseline_mean + 2 * std_val if std_val > 0 else baseline_mean * 1.5
    spikes = [v for v in values_arr if v > threshold]
    spike_detected = len(spikes) > 0

    # Score confidence based on milestone type
    confidence = _score_iot_result(milestone_type, change_pct, spike_detected, len(all_readings))

    # Interpret
    if confidence >= 0.8:
        interpretation = f"IoT data strongly supports milestone. Change: {change_pct:+.1f}%, spikes: {len(spikes)}"
    elif confidence >= 0.5:
        interpretation = f"Moderate IoT evidence. Change: {change_pct:+.1f}%, spikes: {len(spikes)}"
    elif confidence > 0:
        interpretation = f"Weak IoT evidence. Change: {change_pct:+.1f}%, spikes: {len(spikes)}"
    else:
        interpretation = f"No IoT evidence of milestone activity. Change: {change_pct:+.1f}%"

    return {
        "status": "ANALYZED",
        "confidence": round(confidence, 3),
        "interpretation": interpretation,
        "readings_count": len(all_readings),
        "baseline_mean": round(baseline_mean, 2),
        "recent_mean": round(recent_mean, 2),
        "change_pct": round(change_pct, 1),
        "spike_detected": spike_detected,
        "spike_count": len(spikes),
        "stats": {
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "min": round(min_val, 2),
            "max": round(max_val, 2),
        },
        "devices_queried": len(devices),
    }


def _score_iot_result(
    milestone_type: str,
    change_pct: float,
    spike_detected: bool,
    readings_count: int,
) -> float:
    """Score confidence based on IoT data patterns."""
    if readings_count < 5:
        return 0.1  # Too few readings for meaningful analysis

    scoring = {
        "fertilizer": lambda: 0.9 if (spike_detected and change_pct > 15) else (0.6 if spike_detected else (0.3 if change_pct > 10 else 0.1)),
        "irrigation": lambda: 0.9 if (spike_detected and change_pct > 10) else (0.6 if change_pct > 5 else 0.2),
        "planting": lambda: 0.7 if abs(change_pct) < 20 else 0.4,
        "growth": lambda: 0.8 if (change_pct > 5 and not spike_detected) else 0.4,
        "harvest": lambda: 0.7 if change_pct < -10 else 0.3,
    }

    scorer = scoring.get(milestone_type, lambda: 0.5 if change_pct > 5 else 0.3)
    return scorer()
