"""
ORACLE ML — health/health_score.py
Computes a 0–100 machine health score from the anomaly detection output.

Health score formula:
    health = 100 * (1 - anomaly_score)

With adjustments for:
  - Baseline z-score deviation (penalises large feature drift)
  - Persistence (persistent anomalies push the score lower)
"""

import numpy as np
from typing import Dict, Any

from .health_state import HealthState, classify_health, health_state_color


# How much the z-score deviation penalises health (tunable)
Z_SCORE_PENALTY_WEIGHT = 0.05

# Maximum z-score deviation that applies a penalty
Z_SCORE_CAP = 10.0


def compute_health_score(
    anomaly_score: float,
    z_score_mean: float = 0.0,
    persistence_factor: float = 0.0,
) -> float:
    """
    Compute the machine health score (0–100).

    Args:
        anomaly_score:      Normalized anomaly score from AnomalyDetector [0, 1]
        z_score_mean:       Mean absolute z-score vs baseline (feature drift)
        persistence_factor: 0 = single event, 1 = persistent anomaly

    Returns:
        float in [0, 100]
    """
    # Base score from anomaly score
    base = 100.0 * (1.0 - anomaly_score)

    # Small additional penalty for large z-score drift
    z_capped = min(z_score_mean, Z_SCORE_CAP)
    z_penalty = z_capped * Z_SCORE_PENALTY_WEIGHT * 100.0

    # Persistence penalty — persistent anomalies push toward 0
    persistence_penalty = persistence_factor * 20.0

    score = base - z_penalty - persistence_penalty
    return float(np.clip(score, 0.0, 100.0))


def build_health_result(
    asset_id: str,
    anomaly_score: float,
    z_score_mean: float = 0.0,
    persistence_factor: float = 0.0,
) -> Dict[str, Any]:
    """
    Full health result dict returned to the backend.

    Returns:
        {
            "asset_id":     str,
            "health_score": float (0–100),
            "health_state": "HEALTHY" | "ATTENTION" | "CRITICAL",
            "color":        str (hex),
        }
    """
    score = compute_health_score(anomaly_score, z_score_mean, persistence_factor)
    state = classify_health(score)

    return {
        "asset_id": asset_id,
        "health_score": round(score, 2),
        "health_state": state.value,
        "color": health_state_color(state),
    }
