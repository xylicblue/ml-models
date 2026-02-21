"""Drone imagery analyzer — runs ML model on latest drone image.

Finds the most recent drone image for a farm and runs the plant
counting model to assess crop health.
"""

from typing import Any, Dict, Optional
import os

from app.config import settings
from app.db.supabase_client import get_supabase
from app.models.registry import registry


async def analyze_drone(
    farm_id: str,
    milestone_id: str,
    milestone_type: str,
) -> Dict[str, Any]:
    """Analyze drone imagery for milestone verification.

    Finds the latest drone image for the farm, runs the plant counting model,
    and derives a confidence score based on plant count vs expectations.
    """
    try:
        supabase = get_supabase()
    except RuntimeError:
        return {
            "status": "NO_DATA",
            "confidence": 0.0,
            "interpretation": "Database not configured",
        }

    # Find the latest drone image for this farm
    flights_response = (
        supabase.table("drone_flights")
        .select("id, flight_date")
        .eq("farm_id", farm_id)
        .order("flight_date", desc=True)
        .limit(1)
        .execute()
    )

    if not flights_response.data:
        return {
            "status": "NO_DATA",
            "confidence": 0.0,
            "interpretation": "No drone flights found for this farm",
        }

    flight = flights_response.data[0]
    flight_id = flight["id"]

    # Find the image layer
    layers_response = (
        supabase.table("drone_imagery_layers")
        .select("filename, layer_type")
        .eq("flight_id", flight_id)
        .limit(1)
        .execute()
    )

    if not layers_response.data:
        return {
            "status": "NO_DATA",
            "confidence": 0.0,
            "interpretation": "No imagery layers found for the latest drone flight",
        }

    layer = layers_response.data[0]
    filename = layer["filename"]

    # Construct image path
    image_path = os.path.join(settings.imagery_dir, filename)
    if not os.path.exists(image_path):
        return {
            "status": "NO_DATA",
            "confidence": 0.0,
            "interpretation": f"Drone image file not found: {filename}",
        }

    # Run plant counting model
    model_id = "wheat_plant_counter_v1"
    model = registry.get(model_id)
    if model is None:
        return {
            "status": "NO_DATA",
            "confidence": 0.0,
            "interpretation": f"Plant counting model '{model_id}' not available",
        }

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = registry.ensure_loaded(model_id, device=device)

    result = model.predict(image_path)

    plant_count = result.get("plant_count", 0)
    avg_size = result.get("avg_size_px", 0.0)

    # Derive confidence based on milestone type and plant count
    confidence = _score_drone_result(milestone_type, plant_count, avg_size)

    # Interpret
    if confidence >= 0.8:
        interpretation = f"Drone analysis strongly supports milestone. Detected {plant_count} plants (avg size: {avg_size:.1f}px)"
    elif confidence >= 0.5:
        interpretation = f"Moderate drone evidence. Detected {plant_count} plants (avg size: {avg_size:.1f}px)"
    elif confidence > 0:
        interpretation = f"Weak drone evidence. Detected {plant_count} plants (avg size: {avg_size:.1f}px)"
    else:
        interpretation = f"No drone evidence of milestone completion. Detected {plant_count} plants"

    return {
        "status": "ANALYZED",
        "confidence": round(confidence, 3),
        "interpretation": interpretation,
        "plant_count": plant_count,
        "avg_size_px": round(avg_size, 2),
        "min_size_px": round(result.get("min_size_px", 0), 2),
        "max_size_px": round(result.get("max_size_px", 0), 2),
        "flight_date": flight["flight_date"],
        "image_file": filename,
        "tiles_processed": result.get("tiles_processed", 0),
    }


def _score_drone_result(
    milestone_type: str,
    plant_count: int,
    avg_size: float,
) -> float:
    """Score confidence based on drone analysis results and milestone expectations."""
    # Default expectations by milestone type
    expectations = {
        "planting": {"min_count": 100, "ideal_count": 500},
        "fertilizer": {"min_count": 100, "ideal_count": 300},
        "irrigation": {"min_count": 50, "ideal_count": 200},
        "growth": {"min_count": 200, "ideal_count": 500},
        "harvest": {"min_count": 0, "ideal_count": 50},  # Low count at harvest
    }

    exp = expectations.get(milestone_type, {"min_count": 100, "ideal_count": 500})

    if milestone_type == "harvest":
        # At harvest, fewer plants is expected
        if plant_count <= exp["ideal_count"]:
            return 1.0
        elif plant_count <= exp["ideal_count"] * 3:
            return 0.5
        else:
            return 0.2
    else:
        # For other milestones, more plants = better
        if plant_count >= exp["ideal_count"]:
            return 1.0
        elif plant_count >= exp["min_count"]:
            return 0.5 + 0.5 * (plant_count - exp["min_count"]) / (exp["ideal_count"] - exp["min_count"])
        elif plant_count > 0:
            return 0.3 * plant_count / exp["min_count"]
        else:
            return 0.0
