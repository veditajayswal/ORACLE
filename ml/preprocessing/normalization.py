"""
ORACLE ML — preprocessing/normalization.py
Normalizes sensor channels so that the ML models are not biased
by the physical scale of each sensor (e.g. acceleration vs temperature).
"""

import numpy as np
from typing import Dict, Tuple, Optional
import joblib
import os


class ChannelNormalizer:
    """
    Stores per-channel min/max statistics learned from a machine's
    baseline period and applies min-max normalization.

    Saved alongside each machine's model so that inference uses the
    same scale as training.
    """

    def __init__(self):
        # channel_name -> (min, max)
        self._stats: Dict[str, Tuple[float, float]] = {}

    def fit(self, channels: Dict[str, np.ndarray]) -> "ChannelNormalizer":
        """Learn min/max from the provided channel arrays."""
        for ch, arr in channels.items():
            if len(arr) == 0:
                continue
            self._stats[ch] = (float(np.min(arr)), float(np.max(arr)))
        return self

    def transform(self, channels: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply learned normalization. Unknown channels are returned unchanged."""
        result = {}
        for ch, arr in channels.items():
            if ch not in self._stats:
                result[ch] = arr
                continue
            mn, mx = self._stats[ch]
            rng = mx - mn
            if rng == 0:
                result[ch] = np.zeros_like(arr, dtype=np.float64)
            else:
                result[ch] = (arr.astype(np.float64) - mn) / rng
        return result

    def fit_transform(self, channels: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.fit(channels).transform(channels)

    def save(self, path: str) -> None:
        """Persist the normalizer stats to disk."""
        joblib.dump(self._stats, path)

    @classmethod
    def load(cls, path: str) -> "ChannelNormalizer":
        """Load previously saved normalizer stats."""
        norm = cls()
        norm._stats = joblib.load(path)
        return norm

    def is_fitted(self) -> bool:
        return len(self._stats) > 0


def z_score_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit-variance normalization for a single array.
    Returns the original array unchanged if std is zero.
    """
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mean) / std


def normalize_channels(
    channels: Dict[str, np.ndarray],
    normalizer: Optional[ChannelNormalizer] = None,
) -> Tuple[Dict[str, np.ndarray], ChannelNormalizer]:
    """
    Normalize all channels.
    - If a fitted normalizer is given, apply it (inference mode).
    - If None, fit a new normalizer and return it (training mode).
    """
    if normalizer is None:
        normalizer = ChannelNormalizer()
        normalized = normalizer.fit_transform(channels)
    else:
        normalized = normalizer.transform(channels)
    return normalized, normalizer
