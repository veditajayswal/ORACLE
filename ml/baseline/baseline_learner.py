"""
ORACLE ML — baseline/baseline_learner.py
Learns what "normal" looks like for a specific machine.
The baseline is built from a period of confirmed healthy operation.
"""

import numpy as np
from typing import Dict, List, Optional
import joblib
import os

from .baseline_profile import BaselineProfile
from ..features.feature_pipeline import (
    extract_features_from_windows,
    build_feature_matrix,
)


class BaselineLearner:
    """
    Learns the normal operating profile of ONE machine.

    Workflow:
        1. Feed N windows of healthy sensor data via .add_window()
        2. Call .fit() to compute the baseline statistics
        3. Call .is_ready() to check if enough data was collected
        4. The learned profile is stored in a BaselineProfile
    """

    # Minimum number of windows needed before the baseline is reliable
    MIN_WINDOWS = 30

    def __init__(self, asset_id: str, sample_rate: float = 100.0):
        self.asset_id = asset_id
        self.sample_rate = sample_rate
        self._feature_dicts: List[Dict[str, float]] = []
        self._is_fitted = False
        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_std: Optional[np.ndarray] = None
        self._feature_names: List[str] = []

    def add_window(self, window: Dict[str, np.ndarray]) -> None:
        """
        Accept one time-window of normal data.
        Call this for each window during the baseline learning phase.
        """
        from ..features.feature_pipeline import extract_window_features
        feats = extract_window_features(window, self.sample_rate)
        self._feature_dicts.append(feats)

    def add_windows(self, windows: List[Dict[str, np.ndarray]]) -> None:
        """Bulk add multiple windows."""
        feat_list = extract_features_from_windows(windows, self.sample_rate)
        self._feature_dicts.extend(feat_list)

    def is_ready(self) -> bool:
        """True when enough data has been collected to fit a reliable baseline."""
        return len(self._feature_dicts) >= self.MIN_WINDOWS

    def fit(self) -> "BaselineProfile":
        """
        Compute the mean and standard deviation of each feature
        over all collected normal windows.
        This defines the machine's 'fingerprint' of normal behaviour.
        """
        if not self._feature_dicts:
            raise RuntimeError(f"[{self.asset_id}] No data to fit baseline on.")

        X, self._feature_names = build_feature_matrix(self._feature_dicts)

        self._baseline_mean = np.mean(X, axis=0)
        self._baseline_std = np.std(X, axis=0)
        self._baseline_std[self._baseline_std == 0] = 1e-6  # avoid division by zero
        self._is_fitted = True

        return BaselineProfile(
            asset_id=self.asset_id,
            feature_names=self._feature_names,
            mean=self._baseline_mean,
            std=self._baseline_std,
            n_windows=len(self._feature_dicts),
        )

    def window_count(self) -> int:
        return len(self._feature_dicts)
