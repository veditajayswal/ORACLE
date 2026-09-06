"""
ORACLE ML — tests/test_features.py
Unit tests for feature extraction.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ml.features.time_domain import extract_time_features
from ml.features.frequency_domain import extract_frequency_features
from ml.features.feature_pipeline import extract_window_features, build_feature_matrix


def make_sine(freq: float = 10.0, duration: float = 1.28, sample_rate: float = 100.0) -> np.ndarray:
    """Generate a clean sinusoid for testing."""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float64)


def make_noise(n: int = 128) -> np.ndarray:
    """Generate white noise for testing."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, n).astype(np.float64)


class TestTimeDomainFeatures:

    def test_rms_sine(self):
        arr = make_sine()
        from features.time_domain import rms
        r = rms(arr)
        # RMS of pure sine ≈ amplitude / sqrt(2) ≈ 0.707
        assert 0.68 < r < 0.73, f"RMS of unit sine should be ~0.707, got {r}"

    def test_kurtosis_sine(self):
        arr = make_sine()
        from features.time_domain import kurtosis
        k = kurtosis(arr)
        # Pure sine kurtosis ≈ 1.5
        assert 1.3 < k < 1.7, f"Sine kurtosis should be ~1.5, got {k}"

    def test_kurtosis_noise_vs_impulse(self):
        from features.time_domain import kurtosis
        noise = make_noise(256)
        # Gaussian noise has kurtosis ≈ 3
        k_noise = kurtosis(noise)
        assert 2.0 < k_noise < 5.0, f"Gaussian noise kurtosis should be ~3, got {k_noise}"

        # Impulse train (fault-like) has much higher kurtosis
        impulse = np.zeros(256)
        impulse[0] = 10.0
        impulse[128] = -10.0
        k_impulse = kurtosis(impulse)
        assert k_impulse > k_noise, "Impulse signal should have higher kurtosis than noise"

    def test_extract_time_features_keys(self):
        arr = make_sine()
        feats = extract_time_features(arr)
        expected_keys = {"rms", "mean_abs", "peak", "peak_to_peak", "variance",
                         "std_dev", "skewness", "kurtosis", "crest_factor",
                         "shape_factor", "impulse_factor"}
        assert expected_keys == set(feats.keys())

    def test_all_features_finite(self):
        arr = make_sine()
        feats = extract_time_features(arr)
        for k, v in feats.items():
            assert np.isfinite(v), f"Feature {k} is not finite: {v}"


class TestFrequencyDomainFeatures:

    def test_dominant_frequency(self):
        sine_10hz = make_sine(freq=10.0)
        feats = extract_frequency_features(sine_10hz, sample_rate=100.0)
        dom_freq = feats["dominant_freq"]
        # Should detect ~10 Hz
        assert abs(dom_freq - 10.0) < 2.0, f"Expected ~10 Hz dominant freq, got {dom_freq}"

    def test_spectral_entropy_noise_vs_sine(self):
        sine = make_sine()
        noise = make_noise(128)
        feats_sine  = extract_frequency_features(sine,  100.0)
        feats_noise = extract_frequency_features(noise, 100.0)
        # Noise should have higher entropy than a pure tone
        assert feats_noise["spectral_entropy"] > feats_sine["spectral_entropy"], \
            "Noise should have higher spectral entropy than sine"

    def test_all_features_finite(self):
        arr = make_sine()
        feats = extract_frequency_features(arr, 100.0)
        for k, v in feats.items():
            assert np.isfinite(v), f"Frequency feature {k} is not finite: {v}"


class TestFeaturePipeline:

    def _make_window(self) -> dict:
        sig = make_sine(freq=10.0)
        return {
            "ax": sig,
            "ay": sig * 0.5,
            "az": np.ones(128) * 9.8,
            "vibration": sig * 2.0,
            "temperature": np.ones(128) * 35.0,
        }

    def test_extract_window_features_returns_dict(self):
        window = self._make_window()
        feats = extract_window_features(window)
        assert isinstance(feats, dict)
        assert len(feats) > 10

    def test_build_feature_matrix_shape(self):
        windows = [self._make_window() for _ in range(5)]
        from features.feature_pipeline import extract_features_from_windows
        feat_dicts = extract_features_from_windows(windows)
        X, names = build_feature_matrix(feat_dicts)
        assert X.shape[0] == 5
        assert X.shape[1] == len(names)
        assert np.all(np.isfinite(X))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
