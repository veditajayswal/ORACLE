"""
ORACLE ML — anomaly/anomaly_score.py
Converts the raw Isolation Forest output into a 0–1 anomaly score
that is consistent, interpretable, and suitable for health calculation.

Raw sklearn score_samples output:
  - More negative  → more anomalous
  - Typical range: roughly -0.7 to 0.2

We map this to [0, 1] where:
  0.0 = definitely normal
  1.0 = definitely anomalous
"""

import numpy as np


# Calibrated thresholds — anomaly score above this → ANOMALY
ANOMALY_THRESHOLD = 0.5


def normalize_isolation_score(raw_score: float) -> float:
    """
    Convert sklearn IsolationForest score_samples output to [0, 1].

    sklearn returns values roughly in [-0.7, 0.2].
    We flip and scale so that:
      raw  0.2 (very normal)  → normalized 0.0
      raw -0.7 (very anomalous) → normalized 1.0
    """
    # Clamp to expected range
    raw_score = float(np.clip(raw_score, -0.8, 0.3))
    # Flip: high raw = low anomaly, low raw = high anomaly
    flipped = -raw_score
    # Shift to [0, 1.1]
    shifted = flipped + 0.3
    # Scale to [0, 1]
    normalized = np.clip(shifted / 1.1, 0.0, 1.0)
    return float(normalized)


def score_from_raw_array(raw_scores: np.ndarray) -> np.ndarray:
    """
    Normalize an array of raw Isolation Forest scores element-wise.
    Returns a float64 array in [0, 1].
    """
    return np.array([normalize_isolation_score(s) for s in raw_scores], dtype=np.float64)


def aggregate_window_scores(scores: np.ndarray) -> float:
    """
    Combine anomaly scores from multiple windows into a single score.
    Uses the 90th percentile so that one bad window is detected
    without being drowned out by many normal windows.
    """
    if len(scores) == 0:
        return 0.0
    return float(np.percentile(scores, 90))


def is_anomalous(score: float, threshold: float = ANOMALY_THRESHOLD) -> bool:
    """Return True if the aggregated anomaly score exceeds the threshold."""
    return score >= threshold
