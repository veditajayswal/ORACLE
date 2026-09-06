"""
ORACLE ML — health/health_state.py
Defines the three machine health states and their thresholds.
"""

from enum import Enum


class HealthState(str, Enum):
    """
    Three-level machine health classification.
    Using str mixin so the enum value can be JSON-serialised directly.
    """
    HEALTHY   = "HEALTHY"    # score >= 70
    ATTENTION = "ATTENTION"  # score 40–69
    CRITICAL  = "CRITICAL"   # score < 40


# Score thresholds
HEALTHY_THRESHOLD   = 70.0
ATTENTION_THRESHOLD = 40.0


def classify_health(score: float) -> HealthState:
    """
    Map a numeric health score (0–100) to a HealthState.

    Args:
        score: float in [0, 100]
    """
    if score >= HEALTHY_THRESHOLD:
        return HealthState.HEALTHY
    elif score >= ATTENTION_THRESHOLD:
        return HealthState.ATTENTION
    else:
        return HealthState.CRITICAL


def health_state_color(state: HealthState) -> str:
    """Return a hex color code for display in the Android app."""
    return {
        HealthState.HEALTHY:   "#4CAF50",  # green
        HealthState.ATTENTION: "#FF9800",  # orange
        HealthState.CRITICAL:  "#F44336",  # red
    }[state]
