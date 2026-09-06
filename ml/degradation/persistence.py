"""
ORACLE ML — degradation/persistence.py
Detects when an anomaly is PERSISTENT across multiple consecutive
observation windows — a key signal that distinguishes real failure
from a transient noise spike.
"""

from collections import deque
from typing import Deque


class PersistenceTracker:
    """
    Tracks whether anomaly has been present across the last N windows.

    A single anomaly score spike could be noise.
    The same score persisting for 5+ windows is a real problem.
    """

    def __init__(self, window_count: int = 5, anomaly_threshold: float = 0.5):
        """
        Args:
            window_count:      How many consecutive windows to look at
            anomaly_threshold: Score above this is considered anomalous
        """
        self.window_count = window_count
        self.anomaly_threshold = anomaly_threshold
        self._history: Deque[float] = deque(maxlen=window_count)

    def update(self, anomaly_score: float) -> None:
        """Add the latest anomaly score to the history."""
        self._history.append(anomaly_score)

    def persistence_ratio(self) -> float:
        """
        Fraction of recent windows that were anomalous.
        0.0 = no anomalies recently
        1.0 = all recent windows were anomalous (persistent fault)
        """
        if not self._history:
            return 0.0
        anomalous = sum(1 for s in self._history if s >= self.anomaly_threshold)
        return anomalous / len(self._history)

    def is_persistent(self, threshold_ratio: float = 0.6) -> bool:
        """
        True when the majority of recent windows were anomalous.

        Args:
            threshold_ratio: Fraction of anomalous windows required (default 60%)
        """
        return self.persistence_ratio() >= threshold_ratio

    def consecutive_anomalous(self) -> int:
        """
        Number of consecutive anomalous windows at the END of the history.
        e.g. [normal, anomaly, anomaly, anomaly] → 3
        """
        count = 0
        for score in reversed(list(self._history)):
            if score >= self.anomaly_threshold:
                count += 1
            else:
                break
        return count

    def reset(self) -> None:
        """Call this when the machine returns to normal."""
        self._history.clear()
