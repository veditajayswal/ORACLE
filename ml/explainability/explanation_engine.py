"""
ORACLE ML — explainability/explanation_engine.py
Selects and assembles the relevant explanation strings
based on the current machine state and what has changed from baseline.
"""

from typing import List, Dict, Any, Optional

from ..baseline.fingerprint import MachineFingerprint
from .explanation_templates import (
    VIBRATION_RMS_THRESHOLD_PCT,
    FREQUENCY_SHIFT_THRESHOLD_PCT,
    KURTOSIS_THRESHOLD_PCT,
    TEMPERATURE_THRESHOLD_PCT,
    ENTROPY_THRESHOLD_PCT,
    vibration_above_baseline,
    frequency_shifted,
    high_kurtosis,
    temperature_elevated,
    persistent_anomaly,
    spectral_entropy_elevated,
    similar_past_event_exists,
    degrading_trend,
)


class ExplanationEngine:
    """
    Generates the list of human-readable reasons behind ORACLE's prediction.
    One instance per machine analysis cycle.
    """

    def __init__(self, fingerprint: MachineFingerprint):
        self._fingerprint = fingerprint

    def explain(
        self,
        current_features: Dict[str, float],
        is_anomaly: bool,
        consecutive_anomalous: int = 0,
        trend_slope: float = 0.0,
        trend_direction: str = "STABLE",
        has_similar_history: bool = False,
    ) -> List[str]:
        """
        Build ordered list of explanation strings.
        Only includes explanations that are actually triggered.

        Args:
            current_features:    dict of feature_name -> float (from feature_pipeline)
            is_anomaly:          whether anomaly was detected
            consecutive_anomalous: consecutive anomalous cycles
            trend_slope:         health trend slope
            trend_direction:     'DEGRADING' | 'STABLE' | 'IMPROVING'
            has_similar_history: whether similar past event found in memory

        Returns:
            List of explanation strings (may be empty if machine is healthy)
        """
        explanations: List[str] = []

        if not is_anomaly and trend_direction != "DEGRADING":
            return explanations  # machine is fine, nothing to explain

        fp = self._fingerprint

        # Get current key values from features
        curr_vib_rms  = current_features.get("vibration_rms", 0.0)
        curr_dom_freq = current_features.get("vibration_dominant_freq", 0.0)
        curr_kurtosis = current_features.get("vibration_kurtosis", 3.0)
        curr_temp     = current_features.get("temperature_mean", 0.0)
        curr_entropy  = current_features.get("vibration_spectral_entropy", 0.0)

        # Compute deviations from fingerprint
        devs = fp.deviation_summary(curr_vib_rms, curr_dom_freq, curr_kurtosis)
        vib_dev_pct  = devs.get("vibration_rms_deviation_pct", 0.0)
        freq_dev_pct = devs.get("dominant_freq_deviation_pct", 0.0)
        kurt_dev_pct = devs.get("kurtosis_deviation_pct", 0.0)

        temp_dev_pct = 0.0
        if fp.normal_temperature > 0:
            temp_dev_pct = (curr_temp - fp.normal_temperature) / fp.normal_temperature * 100.0

        entropy_dev_pct = 0.0
        if fp.normal_spectral_entropy > 0:
            entropy_dev_pct = (curr_entropy - fp.normal_spectral_entropy) / fp.normal_spectral_entropy * 100.0

        # 1. Vibration above baseline
        if vib_dev_pct > VIBRATION_RMS_THRESHOLD_PCT:
            explanations.append(vibration_above_baseline(vib_dev_pct))

        # 2. Frequency shift
        if abs(freq_dev_pct) > FREQUENCY_SHIFT_THRESHOLD_PCT:
            explanations.append(frequency_shifted(abs(freq_dev_pct)))

        # 3. High kurtosis (impulsive events)
        if kurt_dev_pct > KURTOSIS_THRESHOLD_PCT and curr_kurtosis > 5.0:
            explanations.append(high_kurtosis())

        # 4. Temperature elevated
        if temp_dev_pct > TEMPERATURE_THRESHOLD_PCT:
            explanations.append(temperature_elevated(temp_dev_pct))

        # 5. Spectral entropy elevated
        if entropy_dev_pct > ENTROPY_THRESHOLD_PCT:
            explanations.append(spectral_entropy_elevated())

        # 6. Persistent anomaly
        if consecutive_anomalous >= 3:
            explanations.append(persistent_anomaly(consecutive_anomalous))

        # 7. Similar history
        if has_similar_history:
            explanations.append(similar_past_event_exists())

        # 8. Degrading trend
        if trend_direction == "DEGRADING" and abs(trend_slope) > 0.5:
            explanations.append(degrading_trend(trend_slope))

        return explanations


def build_explanation_result(
    asset_id: str,
    explanations: List[str],
    recommendation: str,
) -> Dict[str, Any]:
    """
    Build the final explanation dict returned to the backend.
    """
    return {
        "asset_id": asset_id,
        "explanations": explanations,
        "explanation_count": len(explanations),
        "recommendation": recommendation,
    }
