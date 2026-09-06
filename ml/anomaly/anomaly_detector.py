"""
ORACLE ML — anomaly/anomaly_detector.py
Main orchestrator for anomaly detection.
Combines the Isolation Forest model with feature extraction
and produces the final anomaly score for a telemetry payload.
"""

import numpy as np
from typing import Dict, List, Any, Optional

from .isolation_forest import OracleIsolationForest
from .anomaly_score import (
    score_from_raw_array,
    aggregate_window_scores,
    is_anomalous,
)
from ..baseline.baseline_profile import BaselineProfile
from ..features.feature_pipeline import (
    features_to_vector,
    extract_window_features,
)


class AnomalyDetector:
    """
    Per-machine anomaly detection pipeline.

    Usage:
        detector = AnomalyDetector(asset_id="pump_01")
        detector.load(model_path, profile)   # load trained model
        result = detector.detect(windows)    # run on new windows
    """

    def __init__(self, asset_id: str):
        self.asset_id = asset_id
        self._model: Optional[OracleIsolationForest] = None
        self._profile: Optional[BaselineProfile] = None

    def load(self, model_path: str, profile: BaselineProfile) -> None:
        """Load a trained model and its corresponding baseline profile."""
        self._model = OracleIsolationForest.load(model_path)
        self._profile = profile

    def is_ready(self) -> bool:
        return self._model is not None and self._profile is not None

    def _windows_to_matrix(
        self, windows: List[Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Extract features from each window and stack into a matrix.
        Uses the feature_names ordering from the baseline profile
        to guarantee consistent column order.
        """
        rows = []
        for w in windows:
            feat_dict = extract_window_features(w)
            vec = features_to_vector(feat_dict, self._profile.feature_names)
            rows.append(vec)
        return np.vstack(rows) if rows else np.empty((0, 0))

    def detect(
        self, windows: List[Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Run anomaly detection on a list of sensor windows.

        Returns:
            {
                "asset_id":      str,
                "anomaly_score": float (0–1),
                "is_anomaly":    bool,
                "window_scores": list of float (one per window),
                "z_score_mean":  float  (mean z-score vs baseline),
                "z_score_max":   float  (worst feature deviation),
            }
        """
        if not self.is_ready():
            raise RuntimeError(
                f"[{self.asset_id}] AnomalyDetector not loaded. Call .load() first."
            )

        if not windows:
            return self._empty_result()

        X = self._windows_to_matrix(windows)
        if X.size == 0:
            return self._empty_result()

        # Raw scores from Isolation Forest (one per window)
        raw = self._model.raw_scores(X)
        window_scores = score_from_raw_array(raw)

        # Aggregate to single score
        agg_score = aggregate_window_scores(window_scores)

        # Z-scores vs baseline (use the mean window feature vector)
        mean_feat_vec = np.mean(X, axis=0)
        z_scores = self._profile.z_scores(mean_feat_vec)

        return {
            "asset_id": self.asset_id,
            "anomaly_score": round(agg_score, 4),
            "is_anomaly": is_anomalous(agg_score),
            "window_scores": [round(float(s), 4) for s in window_scores],
            "z_score_mean": round(float(np.mean(np.abs(z_scores))), 4),
            "z_score_max": round(float(np.max(np.abs(z_scores))), 4),
            "z_scores_by_feature": {
                name: round(float(z), 4)
                for name, z in zip(self._profile.feature_names, z_scores)
            },
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "anomaly_score": 0.0,
            "is_anomaly": False,
            "window_scores": [],
            "z_score_mean": 0.0,
            "z_score_max": 0.0,
            "z_scores_by_feature": {},
        }
