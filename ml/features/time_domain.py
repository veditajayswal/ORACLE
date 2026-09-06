"""
ORACLE ML — features/time_domain.py
Extracts statistical features from a sensor signal window.
These features describe the signal's amplitude distribution and energy.
"""

import numpy as np
from typing import Dict


def rms(arr: np.ndarray) -> float:
    """Root Mean Square — overall vibration energy level."""
    return float(np.sqrt(np.mean(arr ** 2)))


def mean_absolute(arr: np.ndarray) -> float:
    """Mean of the absolute values."""
    return float(np.mean(np.abs(arr)))


def peak_value(arr: np.ndarray) -> float:
    """Maximum absolute value in the window."""
    return float(np.max(np.abs(arr)))


def peak_to_peak(arr: np.ndarray) -> float:
    """Difference between the maximum and minimum value."""
    return float(np.max(arr) - np.min(arr))


def variance(arr: np.ndarray) -> float:
    """Variance of the signal."""
    return float(np.var(arr))


def std_dev(arr: np.ndarray) -> float:
    """Standard deviation of the signal."""
    return float(np.std(arr))


def skewness(arr: np.ndarray) -> float:
    """
    Measure of asymmetry of the distribution.
    Healthy machines often have near-zero skewness.
    """
    n = len(arr)
    if n < 3:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


def kurtosis(arr: np.ndarray) -> float:
    """
    Measure of the 'tailedness' of the distribution.
    Impulsive faults (bearing defects) cause high kurtosis.
    Healthy machines have kurtosis close to 3.
    """
    n = len(arr)
    if n < 4:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 4))


def crest_factor(arr: np.ndarray) -> float:
    """
    Peak / RMS.
    High crest factor = impulsive shock events.
    """
    r = rms(arr)
    if r == 0:
        return 0.0
    return float(peak_value(arr) / r)


def shape_factor(arr: np.ndarray) -> float:
    """RMS / mean_absolute. Sensitive to waveform shape changes."""
    ma = mean_absolute(arr)
    if ma == 0:
        return 0.0
    return float(rms(arr) / ma)


def impulse_factor(arr: np.ndarray) -> float:
    """Peak / mean_absolute. Detects isolated impulse events."""
    ma = mean_absolute(arr)
    if ma == 0:
        return 0.0
    return float(peak_value(arr) / ma)


def extract_time_features(arr: np.ndarray) -> Dict[str, float]:
    """
    Compute all time-domain features for a single 1-D window.

    Returns:
        dict of feature_name -> float value
    """
    return {
        "rms":            rms(arr),
        "mean_abs":       mean_absolute(arr),
        "peak":           peak_value(arr),
        "peak_to_peak":   peak_to_peak(arr),
        "variance":       variance(arr),
        "std_dev":        std_dev(arr),
        "skewness":       skewness(arr),
        "kurtosis":       kurtosis(arr),
        "crest_factor":   crest_factor(arr),
        "shape_factor":   shape_factor(arr),
        "impulse_factor": impulse_factor(arr),
    }
