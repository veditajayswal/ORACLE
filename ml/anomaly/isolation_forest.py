"""
ORACLE ML — anomaly/isolation_forest.py
Wraps scikit-learn's IsolationForest for ORACLE.

Why Isolation Forest?
- Unsupervised — no need for labeled fault data
- Works on high-dimensional feature vectors
- Fast inference
- Naturally handles multivariate sensor data
"""

import numpy as np
from typing import List, Optional
import joblib
import os

from sklearn.ensemble import IsolationForest


class OracleIsolationForest:
    """
    Per-machine Isolation Forest anomaly detector.
    Trained on normal baseline feature vectors.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,  # assume 5% of baseline data might be slightly off
        random_state: int = 42,
    ):
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            warm_start=False,
        )
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "OracleIsolationForest":
        """
        Train on a 2-D matrix of normal feature vectors.
        X shape: (n_windows, n_features)
        """
        if X.ndim != 2 or X.shape[0] < 10:
            raise ValueError(
                f"Expected 2-D matrix with at least 10 rows. Got shape {X.shape}"
            )
        self._model.fit(X)
        self._is_fitted = True
        return self

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw anomaly scores from sklearn.
        sklearn's score_samples returns negative — more negative = more anomalous.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._model.score_samples(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return +1 (normal) or -1 (anomaly) labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._model.predict(X)

    def is_anomaly(self, X: np.ndarray) -> bool:
        """
        True if a single feature vector is classified as anomalous.
        """
        pred = self.predict(X.reshape(1, -1))
        return bool(pred[0] == -1)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: str) -> "OracleIsolationForest":
        detector = cls()
        detector._model = joblib.load(path)
        detector._is_fitted = True
        return detector
