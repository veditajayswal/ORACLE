"""
ORACLE ML — features/feature_pipeline.py
Wires together the full feature extraction pipeline.
Takes a time-window (dict of channel arrays) and produces
a flat feature vector ready for the ML models.
"""

import numpy as np
from typing import Dict, List

from .time_domain import extract_time_features
from .frequency_domain import extract_frequency_features


# Which channels get time-domain features
TIME_DOMAIN_CHANNELS = ["vibration", "ax", "ay", "az"]

# Which channels get frequency-domain features
FREQUENCY_DOMAIN_CHANNELS = ["vibration", "ax"]

# Sample rate (Hz) — must match the ESP32 firmware setting
SAMPLE_RATE = 100.0


def extract_window_features(
    window: Dict[str, np.ndarray],
    sample_rate: float = SAMPLE_RATE,
) -> Dict[str, float]:
    """
    Extract all features from one time-window.

    Args:
        window:      dict of channel_name -> 1-D ndarray (one window)
        sample_rate: sensor sampling rate in Hz

    Returns:
        Flat dict of feature_name -> float
    """
    features: Dict[str, float] = {}

    # Time-domain features for each specified channel
    for ch in TIME_DOMAIN_CHANNELS:
        arr = window.get(ch)
        if arr is None or len(arr) == 0:
            continue
        ch_feats = extract_time_features(arr)
        for feat_name, feat_val in ch_feats.items():
            features[f"{ch}_{feat_name}"] = feat_val

    # Frequency-domain features for each specified channel
    for ch in FREQUENCY_DOMAIN_CHANNELS:
        arr = window.get(ch)
        if arr is None or len(arr) == 0:
            continue
        ch_feats = extract_frequency_features(arr, sample_rate=sample_rate)
        for feat_name, feat_val in ch_feats.items():
            features[f"{ch}_{feat_name}"] = feat_val

    # Temperature — single scalar, no windowing needed (keep last value)
    temp_arr = window.get("temperature")
    if temp_arr is not None and len(temp_arr) > 0:
        features["temperature_mean"] = float(np.mean(temp_arr))
        features["temperature_max"] = float(np.max(temp_arr))

    return features


def extract_features_from_windows(
    windows: List[Dict[str, np.ndarray]],
    sample_rate: float = SAMPLE_RATE,
) -> List[Dict[str, float]]:
    """
    Apply feature extraction to each window in the list.
    Returns one feature dict per window.
    """
    return [extract_window_features(w, sample_rate) for w in windows]


def features_to_vector(feature_dict: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    """
    Convert a feature dict to a 1-D numpy array using a fixed ordering.
    NaN values are replaced with 0.

    Args:
        feature_dict:  output from extract_window_features
        feature_names: ordered list of feature keys (from baseline fitting)
    """
    vec = np.array([feature_dict.get(name, 0.0) for name in feature_names], dtype=np.float64)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    return vec


def build_feature_matrix(
    feature_dicts: List[Dict[str, float]],
) -> tuple[np.ndarray, List[str]]:
    """
    Stack a list of feature dicts into a 2-D matrix (n_windows x n_features).
    Also returns the ordered feature name list for consistent ordering later.

    Returns:
        X:             (n_windows, n_features) float64 matrix
        feature_names: list of feature names (column order)
    """
    if not feature_dicts:
        return np.empty((0, 0)), []

    # Collect all possible feature names (sorted for determinism)
    all_names = sorted({k for d in feature_dicts for k in d.keys()})

    rows = []
    for d in feature_dicts:
        row = np.array([d.get(name, 0.0) for name in all_names], dtype=np.float64)
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        rows.append(row)

    X = np.vstack(rows)
    return X, all_names
