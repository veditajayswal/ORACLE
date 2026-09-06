"""
ORACLE ML — tests/test_health.py
Tests for health scoring, degradation, risk, and explainability.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from ml.health.health_score import compute_health_score, build_health_result
from ml.health.health_state import classify_health, HealthState
from ml.degradation.persistence import PersistenceTracker
from ml.degradation.trend import HealthTrendAnalyzer
from ml.degradation.degradation_tracker import DegradationTracker
from ml.risk.risk_predictor import compute_risk_probability, classify_risk
from ml.risk.confidence import compute_confidence
from ml.risk.recommendation import get_recommendation, should_trigger_alert, should_command_stop


class TestHealthScore:

    def test_healthy_score(self):
        score = compute_health_score(anomaly_score=0.0)
        assert score == 100.0

    def test_critical_score(self):
        score = compute_health_score(anomaly_score=1.0)
        assert score == 0.0

    def test_moderate_score(self):
        score = compute_health_score(anomaly_score=0.5)
        assert 30.0 < score < 70.0

    def test_persistence_penalty(self):
        score_no_persistence = compute_health_score(0.5, persistence_factor=0.0)
        score_with_persistence = compute_health_score(0.5, persistence_factor=1.0)
        assert score_with_persistence < score_no_persistence

    def test_score_clamped_to_0_100(self):
        s1 = compute_health_score(anomaly_score=2.0)  # too high
        s2 = compute_health_score(anomaly_score=-1.0) # too low
        assert 0.0 <= s1 <= 100.0
        assert 0.0 <= s2 <= 100.0


class TestHealthState:

    def test_classify_healthy(self):
        assert classify_health(90.0) == HealthState.HEALTHY

    def test_classify_attention(self):
        assert classify_health(55.0) == HealthState.ATTENTION

    def test_classify_critical(self):
        assert classify_health(20.0) == HealthState.CRITICAL

    def test_boundary_healthy(self):
        assert classify_health(70.0) == HealthState.HEALTHY
        assert classify_health(69.9) == HealthState.ATTENTION

    def test_boundary_critical(self):
        assert classify_health(40.0) == HealthState.ATTENTION
        assert classify_health(39.9) == HealthState.CRITICAL


class TestPersistenceTracker:

    def test_no_anomalies(self):
        pt = PersistenceTracker(window_count=5)
        for _ in range(5):
            pt.update(0.1)
        assert not pt.is_persistent()
        assert pt.persistence_ratio() == 0.0

    def test_all_anomalies(self):
        pt = PersistenceTracker(window_count=5)
        for _ in range(5):
            pt.update(0.9)
        assert pt.is_persistent()
        assert pt.persistence_ratio() == 1.0

    def test_consecutive_count(self):
        pt = PersistenceTracker(window_count=5)
        pt.update(0.1)   # normal
        pt.update(0.9)   # anomaly
        pt.update(0.9)   # anomaly
        pt.update(0.9)   # anomaly
        assert pt.consecutive_anomalous() == 3


class TestHealthTrend:

    def test_degrading_trend(self):
        trend = HealthTrendAnalyzer(history_size=10)
        for score in [90, 85, 80, 75, 70, 65, 60]:
            trend.update(score)
        assert trend.direction() == "DEGRADING"
        assert trend.slope() < 0

    def test_improving_trend(self):
        trend = HealthTrendAnalyzer(history_size=10)
        for score in [50, 55, 60, 65, 70, 75, 80]:
            trend.update(score)
        assert trend.direction() == "IMPROVING"
        assert trend.slope() > 0

    def test_projection_clamped(self):
        trend = HealthTrendAnalyzer()
        for score in [90, 85, 80, 75, 70, 65, 60, 55, 50, 45]:
            trend.update(score)
        proj = trend.projected_health(steps=100)
        assert 0.0 <= proj <= 100.0


class TestRiskPredictor:

    def test_low_risk_healthy(self):
        prob = compute_risk_probability(
            health_score=90.0, trend_slope=0.0,
            persistence_ratio=0.0, consecutive_anomalous=0
        )
        assert classify_risk(prob) in ("LOW", "MEDIUM")

    def test_critical_risk(self):
        prob = compute_risk_probability(
            health_score=10.0, trend_slope=-5.0,
            persistence_ratio=1.0, consecutive_anomalous=10
        )
        assert classify_risk(prob) in ("HIGH", "CRITICAL")

    def test_probability_clamped(self):
        prob = compute_risk_probability(100.0, 10.0, 1.0, 20)
        assert 0.0 <= prob <= 1.0


class TestRecommendations:

    def test_stop_command_only_critical(self):
        assert should_command_stop("CRITICAL", "CRITICAL") is True
        assert should_command_stop("HIGH", "CRITICAL") is False
        assert should_command_stop("CRITICAL", "ATTENTION") is False

    def test_alert_triggered_for_high_risk(self):
        assert should_trigger_alert("HIGH", "HEALTHY") is True
        assert should_trigger_alert("LOW", "HEALTHY") is False

    def test_recommendation_not_empty(self):
        rec = get_recommendation("CRITICAL", "CRITICAL")
        assert len(rec) > 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
