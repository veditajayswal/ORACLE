"""
ORACLE ML — degradation/degradation_tracker.py
Combines PersistenceTracker and HealthTrendAnalyzer into one
per-machine degradation monitor.
"""

import numpy as np
from typing import Dict, Any

from .persistence import PersistenceTracker
from .trend import HealthTrendAnalyzer


class DegradationTracker:
    """
    One instance per machine.
    Must be kept alive between telemetry cycles (in memory or serialised to backend).
    """

    def __init__(
        self,
        asset_id: str,
        persistence_window: int = 5,
        trend_history: int = 20,
        anomaly_threshold: float = 0.5,
    ):
        self.asset_id = asset_id
        self._persistence = PersistenceTracker(
            window_count=persistence_window,
            anomaly_threshold=anomaly_threshold,
        )
        self._trend = HealthTrendAnalyzer(history_size=trend_history)
        self._cycle_count: int = 0

    def update(self, anomaly_score: float, health_score: float) -> None:
        """
        Call once per analysis cycle with the latest scores.
        """
        self._persistence.update(anomaly_score)
        self._trend.update(health_score)
        self._cycle_count += 1

    def persistence_factor(self) -> float:
        """
        0.0 to 1.0 — how persistent the anomaly is.
        Used by health_score.py to apply the persistence penalty.
        """
        return self._persistence.persistence_ratio()

    def is_persistent_anomaly(self) -> bool:
        return self._persistence.is_persistent()

    def consecutive_anomalous_cycles(self) -> int:
        return self._persistence.consecutive_anomalous()

    def trend_direction(self) -> str:
        return self._trend.direction()

    def trend_slope(self) -> float:
        return self._trend.slope()

    def projected_health(self, steps: int = 10) -> float:
        return self._trend.projected_health(steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "cycle_count": self._cycle_count,
            "persistence_ratio": round(self.persistence_factor(), 4),
            "is_persistent_anomaly": self.is_persistent_anomaly(),
            "consecutive_anomalous_cycles": self.consecutive_anomalous_cycles(),
            **self._trend.to_dict(),
        }
