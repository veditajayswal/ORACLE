"""
ORACLE ML — baseline/baseline_profile.py
Data class that holds the learned normal profile for a machine.
Can be saved to and loaded from disk.
"""

import numpy as np
from typing import List, Dict
import joblib
import os


class BaselineProfile:
    """
    Immutable snapshot of a machine's normal operating statistics.
    Stored alongside the trained anomaly model for this machine.
    """

    def __init__(
        self,
        asset_id: str,
        feature_names: List[str],
        mean: np.ndarray,
        std: np.ndarray,
        n_windows: int,
    ):
        self.asset_id = asset_id
        self.feature_names = feature_names
        self.mean = mean
        self.std = std
        self.n_windows = n_windows

    def z_scores(self, feature_vec: np.ndarray) -> np.ndarray:
        """
        Return the per-feature z-score of a new feature vector against the baseline.
        z-score = (value - baseline_mean) / baseline_std
        High z-scores indicate deviation from normal.
        """
        return (feature_vec - self.mean) / self.std

    def max_z_score(self, feature_vec: np.ndarray) -> float:
        """Return the largest z-score (worst deviation) across all features."""
        return float(np.max(np.abs(self.z_scores(feature_vec))))

    def mean_z_score(self, feature_vec: np.ndarray) -> float:
        """Return the mean absolute z-score across all features."""
        return float(np.mean(np.abs(self.z_scores(feature_vec))))

    def to_dict(self) -> Dict:
        """Serialisable summary (for sending to backend if needed)."""
        return {
            "asset_id": self.asset_id,
            "n_windows": self.n_windows,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
        }

    def save(self, path: str) -> None:
        """Persist the profile to disk using joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaselineProfile":
        """Load a previously saved profile."""
        return joblib.load(path)
