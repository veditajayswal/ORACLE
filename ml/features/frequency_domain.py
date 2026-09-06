"""
ORACLE ML — features/frequency_domain.py
Extracts frequency-domain features from a sensor signal window using FFT.
These capture changes in the machine's vibration frequency signature
that time-domain features often miss.
"""

import numpy as np
from typing import Dict


def compute_fft(arr: np.ndarray, sample_rate: float = 100.0):
    """
    Compute the single-sided FFT magnitude spectrum of a 1-D signal.

    Args:
        arr:         1-D signal window
        sample_rate: Sampling rate in Hz (default 100 Hz for ESP32 setup)

    Returns:
        freqs: frequency axis (Hz)
        magnitudes: magnitude spectrum (non-negative frequencies only)
    """
    n = len(arr)
    fft_vals = np.fft.rfft(arr)
    magnitudes = np.abs(fft_vals) / n  # normalize by window length
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    return freqs, magnitudes


def dominant_frequency(freqs: np.ndarray, magnitudes: np.ndarray) -> float:
    """
    Frequency bin with the highest magnitude (peak frequency).
    A shift in this value from baseline indicates a changed machine state.
    """
    if len(magnitudes) == 0:
        return 0.0
    idx = np.argmax(magnitudes)
    return float(freqs[idx])


def spectral_centroid(freqs: np.ndarray, magnitudes: np.ndarray) -> float:
    """
    Weighted average of the frequency spectrum — the 'centre of mass'.
    Healthy machines have a stable centroid; degradation shifts it.
    """
    total = np.sum(magnitudes)
    if total == 0:
        return 0.0
    return float(np.sum(freqs * magnitudes) / total)


def spectral_energy(magnitudes: np.ndarray) -> float:
    """Total power in the spectrum (sum of squared magnitudes)."""
    return float(np.sum(magnitudes ** 2))


def spectral_entropy(magnitudes: np.ndarray) -> float:
    """
    Shannon entropy of the normalized power spectrum.
    Low entropy = energy concentrated at a few frequencies (structured signal).
    High entropy = energy spread widely (chaotic / noisy / faulty).
    """
    power = magnitudes ** 2
    total = np.sum(power)
    if total == 0:
        return 0.0
    prob = power / total
    # avoid log(0)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def spectral_flatness(magnitudes: np.ndarray) -> float:
    """
    Ratio of geometric mean to arithmetic mean of the magnitude spectrum.
    Value near 1 = white noise; near 0 = tonal / structured signal.
    """
    if len(magnitudes) == 0:
        return 0.0
    geo_mean = np.exp(np.mean(np.log(magnitudes + 1e-10)))
    arith_mean = np.mean(magnitudes)
    if arith_mean == 0:
        return 0.0
    return float(geo_mean / arith_mean)


def band_energy_ratio(
    freqs: np.ndarray,
    magnitudes: np.ndarray,
    low: float = 0.0,
    high: float = 50.0,
) -> float:
    """
    Fraction of total spectral energy in [low, high] Hz band.
    Useful for detecting energy migration between frequency bands.
    """
    total = spectral_energy(magnitudes)
    if total == 0:
        return 0.0
    mask = (freqs >= low) & (freqs <= high)
    band = np.sum(magnitudes[mask] ** 2)
    return float(band / total)


def extract_frequency_features(
    arr: np.ndarray,
    sample_rate: float = 100.0,
) -> Dict[str, float]:
    """
    Compute all frequency-domain features for a single 1-D window.

    Returns:
        dict of feature_name -> float value
    """
    freqs, mags = compute_fft(arr, sample_rate)
    return {
        "dominant_freq":        dominant_frequency(freqs, mags),
        "spectral_centroid":    spectral_centroid(freqs, mags),
        "spectral_energy":      spectral_energy(mags),
        "spectral_entropy":     spectral_entropy(mags),
        "spectral_flatness":    spectral_flatness(mags),
        "band_energy_low":      band_energy_ratio(freqs, mags, 0.0, 25.0),
        "band_energy_high":     band_energy_ratio(freqs, mags, 25.0, 50.0),
    }
