"""
ORACLE ML — tests/test_anomaly.py
Tests for the full anomaly detection pipeline.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ml.baseline.baseline_learner import BaselineLearner
from ml.anomaly.isolation_forest import OracleIsolationForest
from ml.anomaly.anomaly_score import normalize_isolation_score, aggregate_window_scores
from ml.anomaly.anomaly_detector import AnomalyDetector
from ml.features.feature_pipeline import build_feature_matrix


def _make_normal_window(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = 128
    return {
        "ax": rng.normal(0, 0.1, n),
        "ay": rng.normal(0, 0.1, n),
        "az": rng.normal(9.8, 0.05, n),
        "vibration": rng.normal(0.5, 0.05, n),
        "temperature": np.ones(n) * 35.0,
    }


def _make_anomalous_window(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = 128
    return {
        "ax": rng.normal(5.0, 2.0, n),
        "ay": rng.normal(5.0, 2.0, n),
        "az": rng.normal(9.8, 3.0, n),
        "vibration": rng.normal(10.0, 2.0, n),
        "temperature": np.ones(n) * 80.0,
    }


def _train_model(n_windows: int = 50):
    """Helper: train a baseline and isolation forest on normal data."""
    learner = BaselineLearner("test_asset")
    for i in range(n_windows):
        learner.add_window(_make_normal_window(i))
    profile = learner.fit()

    X, _ = build_feature_matrix(learner._feature_dicts)
    model = OracleIsolationForest(n_estimators=50, contamination=0.05)
    model.fit(X)
    return model, profile


class TestAnomalyScore:

    def test_normalize_score_range(self):
        for raw in [-0.8, -0.5, -0.3, 0.0, 0.2]:
            norm = normalize_isolation_score(raw)
            assert 0.0 <= norm <= 1.0, f"Normalized score out of range for raw={raw}: {norm}"

    def test_normal_data_low_score(self):
        """Raw scores from sklearn for inlier data are typically > -0.3 → normalized < 0.5"""
        model, _ = _train_model()
        from features.feature_pipeline import extract_window_features, features_to_vector
        window = _make_normal_window(seed=999)
        feat = extract_window_features(window)
        # We need feature names from learner
        learner = BaselineLearner("t")
        for i in range(50):
            learner.add_window(_make_normal_window(i))
        profile = learner.fit()
        vec = features_to_vector(feat, profile.feature_names)
        raw = model.raw_scores(vec.reshape(1, -1))[0]
        norm = normalize_isolation_score(raw)
        # Normal data should have low anomaly score
        assert norm < 0.7, f"Normal data should have low anomaly score, got {norm}"

    def test_aggregate_uses_90th_percentile(self):
        # Mix of mostly normal scores with one spike
        scores = np.array([0.1, 0.1, 0.2, 0.1, 0.9])
        agg = aggregate_window_scores(scores)
        assert agg > 0.5, "The spike should be captured in 90th percentile"


class TestAnomalyDetector:

    def _build_detector(self):
        learner = BaselineLearner("asset_x")
        for i in range(50):
            learner.add_window(_make_normal_window(i))
        profile = learner.fit()

        X, _ = build_feature_matrix(learner._feature_dicts)
        iforest = OracleIsolationForest(n_estimators=50)
        iforest.fit(X)

        # Save temp files for detector.load()
        import tempfile, joblib
        tmpdir = tempfile.mkdtemp()
        model_path = os.path.join(tmpdir, "isolation_forest.pkl")
        iforest.save(model_path)

        detector = AnomalyDetector("asset_x")
        detector.load(model_path, profile)
        return detector

    def test_detect_normal_returns_low_score(self):
        detector = self._build_detector()
        normal_windows = [_make_normal_window(seed=i+100) for i in range(3)]
        result = detector.detect(normal_windows)
        assert result["anomaly_score"] < 0.8, \
            f"Normal windows should have low anomaly score, got {result['anomaly_score']}"

    def test_detect_returns_required_keys(self):
        detector = self._build_detector()
        windows = [_make_normal_window(0)]
        result = detector.detect(windows)
        required = {"asset_id", "anomaly_score", "is_anomaly", "window_scores",
                    "z_score_mean", "z_score_max"}
        assert required.issubset(result.keys())

    def test_detect_empty_windows(self):
        detector = self._build_detector()
        result = detector.detect([])
        assert result["anomaly_score"] == 0.0
        assert result["is_anomaly"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
