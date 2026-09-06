"""
ORACLE ML — risk/confidence.py
Calculates ORACLE's confidence in its own prediction.
Higher confidence means the model has seen more data and the signal is clear.
"""

import numpy as np


def compute_confidence(
    anomaly_score: float,
    z_score_max: float,
    n_baseline_windows: int,
    persistence_ratio: float,
) -> float:
    """
    Estimate prediction confidence on a 0–1 scale.

    Factors:
    1. Decisiveness: predictions near 0 or 1 are more confident than 0.5
    2. Feature deviation: large z-score means a clear, unambiguous signal
    3. Baseline size: more training data = more reliable model
    4. Persistence: persistent anomalies are clearer than single spikes

    Returns:
        float in [0, 1]
    """
    # 1. Decisiveness — how far from 0.5 is the anomaly score?
    decisiveness = abs(anomaly_score - 0.5) * 2.0  # 0 at 0.5, 1 at 0 or 1

    # 2. Feature clarity — large z-score = clear deviation from baseline
    z_clarity = min(z_score_max / 5.0, 1.0)  # saturates at z=5

    # 3. Baseline maturity — more windows = higher confidence
    baseline_factor = min(n_baseline_windows / 100.0, 1.0)  # saturates at 100 windows

    # 4. Persistence contribution
    persistence_contribution = persistence_ratio  # already 0–1

    # Weighted average
    confidence = (
        0.35 * decisiveness +
        0.30 * z_clarity +
        0.20 * baseline_factor +
        0.15 * persistence_contribution
    )

    return float(np.clip(confidence, 0.0, 1.0))


def confidence_label(confidence: float) -> str:
    """Human-readable confidence label."""
    if confidence >= 0.8:
        return "HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"
