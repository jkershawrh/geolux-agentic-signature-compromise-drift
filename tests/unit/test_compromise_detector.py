import pytest

from domain.enums import DriftCategory
from domain.geometry import DriftMeasurement
from engine.compromise_detector import CompromiseDetector


def _make_drift(compromise_prob, geo_dist, magnitude, significant=True):
    return DriftMeasurement(
        agent_id="test",
        baseline_signature_id="s1",
        current_signature_id="s2",
        geodesic_distance=geo_dist,
        euclidean_distance=geo_dist * 0.8,
        cosine_similarity=max(0.0, 1.0 - geo_dist),
        drift_category=DriftCategory.GOAL,
        drift_magnitude=magnitude,
        per_dimension_drift={"response_structure": magnitude},
        is_significant=significant,
        p_value=0.01 if significant else 0.8,
        compromise_probability=compromise_prob,
    )


class TestCompromiseDetector:
    def test_no_alert_below_threshold(self):
        detector = CompromiseDetector()
        drift = _make_drift(0.1, 0.1, 0.1, significant=False)
        assert detector.evaluate(drift) is None

    def test_warning_at_threshold(self):
        detector = CompromiseDetector(warning_threshold=0.5)
        drift = _make_drift(0.6, 0.4, 0.6)
        alert = detector.evaluate(drift)
        assert alert is not None
        assert alert.severity == "warning"

    def test_critical_at_threshold(self):
        detector = CompromiseDetector(critical_threshold=0.8)
        drift = _make_drift(0.9, 1.5, 0.9)
        alert = detector.evaluate(drift)
        assert alert is not None
        assert alert.severity == "critical"

    def test_critical_on_high_distance(self):
        detector = CompromiseDetector(distance_critical=1.0)
        drift = _make_drift(0.3, 1.5, 0.3)
        alert = detector.evaluate(drift)
        assert alert is not None
        assert alert.severity == "critical"

    def test_alert_contains_evidence(self):
        detector = CompromiseDetector()
        drift = _make_drift(0.9, 1.5, 0.9)
        alert = detector.evaluate(drift)
        assert "geodesic_distance" in alert.evidence
        assert "p_value" in alert.evidence

    def test_alert_has_recommendation(self):
        detector = CompromiseDetector()
        drift = _make_drift(0.9, 1.5, 0.9)
        alert = detector.evaluate(drift)
        assert "CRITICAL" in alert.recommendation or "WARNING" in alert.recommendation

    def test_evaluate_multiple(self):
        detector = CompromiseDetector()
        drifts = [
            _make_drift(0.01, 0.01, 0.01, significant=False),
            _make_drift(0.9, 1.5, 0.9),
            _make_drift(0.6, 0.5, 0.6),
        ]
        alerts = detector.evaluate_multiple(drifts)
        assert len(alerts) == 2
