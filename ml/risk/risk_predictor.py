"""
ORACLE ML — risk/risk_predictor.py
Estimates failure risk from the current machine state.

Risk is distinct from health score:
  - Health score = current state (how bad is it NOW?)
  - Risk = future probability (how likely is failure SOON?)
"""

import numpy as np
from typing import Dict, Any


# Risk levels
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def compute_risk_probability(
    health_score: float,
    trend_slope: float,
    persistence_ratio: float,
    consecutive_anomalous: int,
) -> float:
    """
    Estimate the probability of failure in the near term.

    Args:
        health_score:           Current health score (0–100)
        trend_slope:            Trend slope (negative = degrading)
        persistence_ratio:      Fraction of recent windows that were anomalous
        consecutive_anomalous:  Count of consecutive anomalous cycles

    Returns:
        float in [0, 1] — failure risk probability
    """
    # 1. Low health = high base risk
    health_risk = 1.0 - (health_score / 100.0)

    # 2. Negative trend amplifies risk
    trend_risk = np.clip(-trend_slope / 10.0, 0.0, 1.0)

    # 3. Persistence is a strong indicator
    persistence_risk = persistence_ratio

    # 4. Consecutive anomalies are a direct failure indicator
    consecutive_risk = np.clip(consecutive_anomalous / 10.0, 0.0, 1.0)

    # Weighted combination
    risk = (
        0.35 * health_risk +
        0.25 * trend_risk +
        0.25 * persistence_risk +
        0.15 * consecutive_risk
    )

    return float(np.clip(risk, 0.0, 1.0))


def classify_risk(probability: float) -> str:
    """
    Map risk probability to a named risk level.

    < 0.25  → LOW
    0.25–0.5 → MEDIUM
    0.5–0.75 → HIGH
    >= 0.75 → CRITICAL
    """
    if probability >= 0.75:
        return "CRITICAL"
    elif probability >= 0.50:
        return "HIGH"
    elif probability >= 0.25:
        return "MEDIUM"
    else:
        return "LOW"


def build_risk_result(
    asset_id: str,
    health_score: float,
    trend_slope: float,
    persistence_ratio: float,
    consecutive_anomalous: int,
    confidence: float,
) -> Dict[str, Any]:
    """
    Build the full risk assessment dict returned to the backend.
    """
    prob = compute_risk_probability(
        health_score, trend_slope, persistence_ratio, consecutive_anomalous
    )
    level = classify_risk(prob)

    return {
        "asset_id": asset_id,
        "risk_probability": round(prob, 4),
        "risk_level": level,
        "confidence": round(confidence, 4),
    }
