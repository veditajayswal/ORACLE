"""
ORACLE ML — baseline/fingerprint.py
Builds a compact, human-inspectable fingerprint of a machine's normal behaviour.
This is what makes ORACLE understand each machine individually.
"""

import numpy as np
from typing import Dict, List, Any

from .baseline_profile import BaselineProfile


class MachineFingerprint:
    """
    A compact, named summary of a machine's normal operating characteristics.

    While BaselineProfile stores the full statistical representation used
    by the ML model, MachineFingerprint stores the key operating metrics
    in interpretable physical units — vibration RMS, dominant frequency, etc.
    This is what gets displayed in the Android app and used for explanations.
    """

    def __init__(self, asset_id: str):
        self.asset_id = asset_id
        # key operating characteristics derived from the baseline
        self.normal_vibration_rms: float = 0.0
        self.normal_dominant_frequency: float = 0.0
        self.normal_temperature: float = 0.0
        self.normal_kurtosis: float = 0.0
        self.normal_spectral_entropy: float = 0.0
        self.established: bool = False

    @classmethod
    def from_baseline_profile(cls, profile: BaselineProfile) -> "MachineFingerprint":
        """
        Extract the fingerprint key values from a fitted BaselineProfile.
        Looks up feature_names to find the relevant baseline means.
        """
        fp = cls(asset_id=profile.asset_id)

        feature_idx = {name: i for i, name in enumerate(profile.feature_names)}

        def get_mean(name: str) -> float:
            idx = feature_idx.get(name)
            if idx is None:
                return 0.0
            return float(profile.mean[idx])

        fp.normal_vibration_rms = get_mean("vibration_rms")
        fp.normal_dominant_frequency = get_mean("vibration_dominant_freq")
        fp.normal_temperature = get_mean("temperature_mean")
        fp.normal_kurtosis = get_mean("vibration_kurtosis")
        fp.normal_spectral_entropy = get_mean("vibration_spectral_entropy")
        fp.established = True

        return fp

    def to_dict(self) -> Dict[str, Any]:
        """Return as a JSON-serialisable dict."""
        return {
            "asset_id": self.asset_id,
            "established": self.established,
            "normal_vibration_rms": self.normal_vibration_rms,
            "normal_dominant_frequency_hz": self.normal_dominant_frequency,
            "normal_temperature_celsius": self.normal_temperature,
            "normal_kurtosis": self.normal_kurtosis,
            "normal_spectral_entropy": self.normal_spectral_entropy,
        }

    def deviation_summary(
        self,
        current_vibration_rms: float,
        current_dominant_freq: float,
        current_kurtosis: float,
    ) -> Dict[str, float]:
        """
        Compare current values against the fingerprint.
        Returns relative deviation (%) for each key characteristic.
        Used by the explanation engine.
        """
        def rel_dev(current: float, normal: float) -> float:
            if normal == 0:
                return 0.0
            return float((current - normal) / normal * 100.0)

        return {
            "vibration_rms_deviation_pct": rel_dev(current_vibration_rms, self.normal_vibration_rms),
            "dominant_freq_deviation_pct": rel_dev(current_dominant_freq, self.normal_dominant_frequency),
            "kurtosis_deviation_pct": rel_dev(current_kurtosis, self.normal_kurtosis),
        }
