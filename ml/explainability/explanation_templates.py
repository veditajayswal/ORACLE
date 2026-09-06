"""
ORACLE ML — explainability/explanation_templates.py
Human-readable explanation strings.
ORACLE uses these to tell the user WHY a machine is flagged.
This is a key differentiator — not just "anomaly detected" but WHY.
"""

from typing import List


# Threshold deviations (%) to trigger each explanation
VIBRATION_RMS_THRESHOLD_PCT   = 25.0
FREQUENCY_SHIFT_THRESHOLD_PCT = 15.0
KURTOSIS_THRESHOLD_PCT        = 40.0
TEMPERATURE_THRESHOLD_PCT     = 20.0
ENTROPY_THRESHOLD_PCT         = 30.0


def vibration_above_baseline(deviation_pct: float) -> str:
    return (
        f"Vibration is {deviation_pct:.0f}% above the machine's established normal baseline."
    )


def frequency_shifted(deviation_pct: float) -> str:
    return (
        f"The dominant frequency has shifted {deviation_pct:.0f}% from the established pattern. "
        "This may indicate mechanical looseness, imbalance, or resonance change."
    )


def high_kurtosis() -> str:
    return (
        "Impulsive shock events detected in the vibration signal. "
        "This is a strong indicator of bearing damage or mechanical impact."
    )


def temperature_elevated(deviation_pct: float) -> str:
    return (
        f"Temperature is {deviation_pct:.0f}% above normal operating levels. "
        "This could indicate friction, overload, or inadequate cooling."
    )


def persistent_anomaly(consecutive_cycles: int) -> str:
    return (
        f"Abnormal behavior has persisted across {consecutive_cycles} consecutive "
        "observation cycles. This is not a transient spike."
    )


def spectral_entropy_elevated() -> str:
    return (
        "The frequency spectrum has become significantly more chaotic than normal. "
        "Energy is spreading across multiple unexpected frequency bands."
    )


def similar_past_event_exists() -> str:
    return (
        "A similar anomaly event exists in this machine's history. "
        "Review past maintenance records for context."
    )


def degrading_trend(slope: float) -> str:
    return (
        f"Machine health has been declining at a rate of {abs(slope):.1f} points per cycle. "
        "Continued degradation is expected without intervention."
    )
