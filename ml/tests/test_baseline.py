"""
ORACLE ML — tests/test_baseline.py
Tests for baseline learning and fingerprint generation.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ml.baseline.baseline_learner import BaselineLearner
from ml.baseline.fingerprint import MachineFingerprint


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


class TestBaselineLearner:

    def test_add_windows_and_count(self):
        learner = BaselineLearner("test_machine")
        for i in range(10):
            learner.add_window(_make_normal_window(i))
        assert learner.window_count() == 10

    def test_is_ready_threshold(self):
        learner = BaselineLearner("test_machine")
        assert not learner.is_ready()
        for i in range(BaselineLearner.MIN_WINDOWS):
            learner.add_window(_make_normal_window(i))
        assert learner.is_ready()

    def test_fit_returns_profile(self):
        learner = BaselineLearner("test_machine")
        for i in range(35):
            learner.add_window(_make_normal_window(i))
        profile = learner.fit()
        assert profile.asset_id == "test_machine"
        assert len(profile.feature_names) > 0
        assert profile.mean is not None
        assert profile.std is not None

    def test_profile_z_scores_normal_data(self):
        """Z-scores of new normal data against the baseline should be small."""
        learner = BaselineLearner("test_machine")
        for i in range(35):
            learner.add_window(_make_normal_window(i))
        profile = learner.fit()

        from features.feature_pipeline import extract_window_features, features_to_vector
        normal_window = _make_normal_window(seed=999)
        feats = extract_window_features(normal_window)
        vec = features_to_vector(feats, profile.feature_names)
        z = profile.mean_z_score(vec)
        assert z < 5.0, f"Normal data should have small z-score vs baseline, got {z}"

    def test_profile_z_scores_anomalous_data(self):
        """Z-scores of clearly anomalous data should be large."""
        learner = BaselineLearner("test_machine")
        for i in range(35):
            learner.add_window(_make_normal_window(i))
        profile = learner.fit()

        from features.feature_pipeline import extract_window_features, features_to_vector
        rng = np.random.default_rng(0)
        # Highly anomalous window — very large vibration
        anomalous_window = {
            "ax": rng.normal(5.0, 2.0, 128),   # very different from baseline
            "ay": rng.normal(5.0, 2.0, 128),
            "az": rng.normal(9.8, 3.0, 128),
            "vibration": rng.normal(10.0, 2.0, 128),  # 20x normal
            "temperature": np.ones(128) * 80.0,         # very hot
        }
        feats = extract_window_features(anomalous_window)
        vec = features_to_vector(feats, profile.feature_names)
        z = profile.mean_z_score(vec)
        assert z > 2.0, f"Anomalous data should have large z-score, got {z}"


class TestMachineFingerprint:

    def test_from_baseline_profile(self):
        learner = BaselineLearner("pump_01")
        for i in range(35):
            learner.add_window(_make_normal_window(i))
        profile = learner.fit()
        fp = MachineFingerprint.from_baseline_profile(profile)
        assert fp.asset_id == "pump_01"
        assert fp.established is True

    def test_to_dict_has_required_keys(self):
        learner = BaselineLearner("pump_01")
        for i in range(35):
            learner.add_window(_make_normal_window(i))
        profile = learner.fit()
        fp = MachineFingerprint.from_baseline_profile(profile)
        d = fp.to_dict()
        assert "asset_id" in d
        assert "established" in d
        assert "normal_vibration_rms" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
