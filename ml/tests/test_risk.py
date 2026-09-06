"""
ORACLE ML — tests/test_risk.py
Unit tests for confidence scoring, failure risk prediction, and recommendations.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

from ml.risk.confidence import compute_confidence, confidence_label
from ml.risk.risk_predictor import compute_risk_probability, classify_risk, build_risk_result
from ml.risk.recommendation import get_recommendation, should_trigger_alert, should_command_stop


class TestConfidence:

    def test_confidence_range(self):
        conf = compute_confidence(
            anomaly_score=0.9,
            z_score_max=4.5,
            n_baseline_windows=80,
            persistence_ratio=0.8
        )
        assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

    def test_confidence_labels(self):
        assert confidence_label(0.85) == "HIGH"
        assert confidence_label(0.65) == "MEDIUM"
        assert confidence_label(0.30) == "LOW"

    def test_more_baseline_data_increases_confidence(self):
        conf_low = compute_confidence(0.8, 3.0, n_baseline_windows=10, persistence_ratio=0.5)
        conf_high = compute_confidence(0.8, 3.0, n_baseline_windows=100, persistence_ratio=0.5)
        assert conf_high > conf_low, "More baseline data should yield higher confidence"


class TestRiskPredictor:

    def test_low_risk_for_healthy_machine(self):
        prob = compute_risk_probability(
            health_score=95.0,
            trend_slope=0.0,
            persistence_ratio=0.0,
            consecutive_anomalous=0
        )
        assert prob < 0.25, f"Healthy machine should have low risk prob, got {prob}"
        assert classify_risk(prob) == "LOW"

    def test_critical_risk_for_failing_machine(self):
        prob = compute_risk_probability(
            health_score=15.0,
            trend_slope=-6.0,
            persistence_ratio=1.0,
            consecutive_anomalous=8
        )
        assert prob >= 0.75, f"Failing machine should have critical risk prob, got {prob}"
        assert classify_risk(prob) == "CRITICAL"

    def test_probability_clamped_bounds(self):
        p1 = compute_risk_probability(100.0, 50.0, 0.0, 0)
        p2 = compute_risk_probability(0.0, -50.0, 1.0, 20)
        assert 0.0 <= p1 <= 1.0
        assert 0.0 <= p2 <= 1.0

    def test_build_risk_result_structure(self):
        res = build_risk_result(
            asset_id="pump_01",
            health_score=85.0,
            trend_slope=0.0,
            persistence_ratio=0.0,
            consecutive_anomalous=0,
            confidence=0.88
        )
        assert res["asset_id"] == "pump_01"
        assert "risk_probability" in res
        assert res["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert res["confidence"] == 0.88


class TestRecommendations:

    def test_stop_command_only_when_both_critical(self):
        # Only stop when BOTH health and risk are CRITICAL
        assert should_command_stop("CRITICAL", "CRITICAL") is True
        assert should_command_stop("HIGH", "CRITICAL") is False
        assert should_command_stop("CRITICAL", "ATTENTION") is False
        assert should_command_stop("LOW", "HEALTHY") is False

    def test_alert_triggered_appropriately(self):
        assert should_trigger_alert("HIGH", "HEALTHY") is True
        assert should_trigger_alert("LOW", "ATTENTION") is True
        assert should_trigger_alert("LOW", "HEALTHY") is False

    def test_recommendation_returns_valid_string(self):
        rec = get_recommendation("CRITICAL", "CRITICAL")
        assert "STOP" in rec or "Immediate" in rec
        assert isinstance(rec, str) and len(rec) > 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
