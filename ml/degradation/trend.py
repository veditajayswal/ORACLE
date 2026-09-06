"""
ORACLE ML — degradation/trend.py
Computes the health score trend over recent history.
A downward trend is the early warning signal for predictive maintenance.
"""

import numpy as np
from typing import List, Dict, Any
from collections import deque


class HealthTrendAnalyzer:
    """
    Maintains a rolling window of health scores and computes:
    - Trend direction (improving / stable / degrading)
    - Trend slope (how fast it is changing)
    - Projected health score in N cycles
    """

    def __init__(self, history_size: int = 20):
        self.history_size = history_size
        self._scores: deque = deque(maxlen=history_size)

    def update(self, health_score: float) -> None:
        """Add the latest health score to the rolling history."""
        self._scores.append(float(health_score))

    def slope(self) -> float:
        """
        Linear regression slope over recent health scores.
        Positive = improving. Negative = degrading.
        Returns 0.0 if fewer than 3 data points.
        """
        scores = list(self._scores)
        n = len(scores)
        if n < 3:
            return 0.0
        x = np.arange(n, dtype=np.float64)
        y = np.array(scores, dtype=np.float64)
        # Simple least-squares slope
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def direction(self) -> str:
        """
        Human-readable trend direction.
        'DEGRADING' | 'STABLE' | 'IMPROVING'
        """
        s = self.slope()
        if s < -1.0:
            return "DEGRADING"
        elif s > 1.0:
            return "IMPROVING"
        else:
            return "STABLE"

    def projected_health(self, steps_ahead: int = 10, steps: int = None) -> float:
        """
        Linearly extrapolate current trend to predict health in N steps.
        Clamped to [0, 100].
        """
        n = steps if steps is not None else steps_ahead
        if not self._scores:
            return 100.0
        current = list(self._scores)[-1]
        projected = current + self.slope() * n
        return float(np.clip(projected, 0.0, 100.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend_slope": round(self.slope(), 4),
            "trend_direction": self.direction(),
            "projected_health_10_cycles": round(self.projected_health(10), 2),
            "history_length": len(self._scores),
        }
