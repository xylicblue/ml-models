"""Verification scorer — weighted combination of multi-source analysis."""

from typing import Any, Dict


# Default source weights
DEFAULT_WEIGHTS = {
    "satellite": 0.40,
    "drone": 0.35,
    "iot": 0.25,
}

# Verdict thresholds
COMPLETE_THRESHOLD = 0.75
REVIEW_THRESHOLD = 0.40


def compute_verdict(
    satellite_result: Dict[str, Any],
    drone_result: Dict[str, Any],
    iot_result: Dict[str, Any],
    weights: Dict[str, float] = None,
) -> Dict[str, Any]:
    """Compute weighted verification verdict from multi-source analysis.

    Adjusts weights dynamically when data sources are missing.

    Returns:
        dict with verdict, overall_confidence, recommendation, and per-source details.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    sources = {
        "satellite": satellite_result,
        "drone": drone_result,
        "iot": iot_result,
    }

    # Determine available sources (those with actual data)
    available = {}
    unavailable = {}
    for name, result in sources.items():
        if result.get("status") == "ANALYZED" and result.get("confidence", 0) > 0:
            available[name] = result
        else:
            unavailable[name] = result

    if not available:
        return {
            "verdict": "INSUFFICIENT_EVIDENCE",
            "overall_confidence": 0.0,
            "recommendation": "Unable to verify milestone — no data sources available. "
                              "Ensure satellite monitoring, drone flights, and IoT sensors are active.",
            "sources_available": 0,
            "sources_total": 3,
            "satellite_analysis": satellite_result,
            "drone_analysis": drone_result,
            "iot_analysis": iot_result,
        }

    # Redistribute weights from unavailable sources proportionally
    total_available_weight = sum(weights[name] for name in available)
    adjusted_weights = {}
    for name in available:
        adjusted_weights[name] = weights[name] / total_available_weight

    # Compute weighted average confidence
    overall_confidence = sum(
        available[name]["confidence"] * adjusted_weights[name]
        for name in available
    )

    # Determine verdict
    if overall_confidence >= COMPLETE_THRESHOLD:
        verdict = "MILESTONE_COMPLETE"
    elif overall_confidence >= REVIEW_THRESHOLD:
        verdict = "MANUAL_REVIEW_REQUIRED"
    else:
        verdict = "MILESTONE_FAILED"

    # Generate recommendation
    recommendation = _generate_recommendation(
        verdict, overall_confidence, available, unavailable
    )

    return {
        "verdict": verdict,
        "overall_confidence": round(overall_confidence, 3),
        "recommendation": recommendation,
        "sources_available": len(available),
        "sources_total": 3,
        "weights_used": adjusted_weights,
        "satellite_analysis": satellite_result,
        "drone_analysis": drone_result,
        "iot_analysis": iot_result,
    }


def _generate_recommendation(
    verdict: str,
    confidence: float,
    available: Dict,
    unavailable: Dict,
) -> str:
    """Generate a human-readable recommendation based on the verdict."""
    parts = []

    if verdict == "MILESTONE_COMPLETE":
        sources_str = ", ".join(available.keys())
        parts.append(
            f"All available indicators ({sources_str}) confirm milestone completion "
            f"with {confidence:.0%} confidence. Recommend approving payment."
        )
    elif verdict == "MANUAL_REVIEW_REQUIRED":
        parts.append(
            f"Evidence is inconclusive ({confidence:.0%} confidence). "
            "Manual review recommended before approving payment."
        )
        # Note which sources are weak
        for name, result in available.items():
            if result["confidence"] < 0.5:
                parts.append(f" - {name}: low confidence ({result['confidence']:.0%})")
    else:
        parts.append(
            f"Verification failed ({confidence:.0%} confidence). "
            "Do not approve payment without additional evidence."
        )

    if unavailable:
        missing = ", ".join(unavailable.keys())
        parts.append(f"Missing data from: {missing}.")

    return " ".join(parts)
