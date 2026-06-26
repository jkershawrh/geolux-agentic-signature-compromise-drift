import pytest

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import DriftCategory, SignatureType
from domain.models import AgentProfile
from engine.drift_detector import DriftDetector
from engine.signature_generator import SignatureGenerator


def _make_sig(agent_id, adapter_kwargs=None, n=5):
    kwargs = adapter_kwargs or {}
    adapter = MockInferenceAdapter(**kwargs)
    extractor = DefaultMetricExtractor()
    generator = SignatureGenerator(manifold_method="pca")
    agent = AgentProfile(agent_id=agent_id, display_name=agent_id, model_id="test")
    metrics_per_run = []
    run_ids = []
    for i in range(n):
        run = adapter.execute(agent, f"Prompt {i}")
        metrics_per_run.append(extractor.extract(run))
        run_ids.append(run.run_id)
    return generator.generate(agent_id, metrics_per_run, run_ids, SignatureType.BASELINE)


class TestDriftDetector:
    def test_identical_signatures_low_drift(self):
        sig = _make_sig("test")
        detector = DriftDetector()
        drift = detector.detect(sig, sig)
        assert drift.geodesic_distance == pytest.approx(0.0, abs=1e-6)
        assert drift.drift_magnitude < 0.5
        assert drift.is_significant is False

    def test_different_behavior_produces_drift(self):
        baseline = _make_sig("test")
        perturbed = _make_sig("test", {
            "response_key": "code", "latency_ms": 500,
            "input_tokens": 300, "output_tokens": 200,
            "thinking_tokens": 100, "include_tool_calls": True,
        })
        detector = DriftDetector()
        drift = detector.detect(baseline, perturbed)
        assert drift.geodesic_distance > 0
        assert drift.euclidean_distance > 0
        assert drift.is_significant is True

    def test_drift_category_is_valid(self):
        baseline = _make_sig("test")
        perturbed = _make_sig("test", {"response_key": "reasoning"})
        detector = DriftDetector()
        drift = detector.detect(baseline, perturbed)
        assert drift.drift_category in DriftCategory

    def test_per_dimension_drift_populated(self):
        baseline = _make_sig("test")
        perturbed = _make_sig("test", {"response_key": "code"})
        detector = DriftDetector()
        drift = detector.detect(baseline, perturbed)
        assert len(drift.per_dimension_drift) == 9

    def test_significance_determined_by_thresholds(self):
        """Significance is based on distance threshold, not permutation p-value."""
        baseline = _make_sig("test")

        # Identical signatures: small distance -> not significant
        detector = DriftDetector(significance_distance_threshold=0.3)
        drift_same = detector.detect(baseline, baseline)
        assert drift_same.is_significant is False
        assert drift_same.p_value is None

        # Different behavior: large distance -> significant
        perturbed = _make_sig("test", {
            "response_key": "code", "latency_ms": 500,
            "input_tokens": 300, "output_tokens": 200,
            "thinking_tokens": 100, "include_tool_calls": True,
        })
        drift_diff = detector.detect(baseline, perturbed)
        assert drift_diff.is_significant is True
        assert drift_diff.p_value is None

    def test_compromise_probability_in_range(self):
        baseline = _make_sig("test")
        perturbed = _make_sig("test", {"response_key": "code"})
        detector = DriftDetector()
        drift = detector.detect(baseline, perturbed)
        assert 0.0 <= drift.compromise_probability <= 1.0

    def test_drift_direction_is_unit_or_zero(self):
        import numpy as np
        baseline = _make_sig("test")
        perturbed = _make_sig("test", {"response_key": "code"})
        detector = DriftDetector()
        drift = detector.detect(baseline, perturbed)
        direction = np.array(drift.drift_direction)
        norm = np.linalg.norm(direction)
        assert norm == pytest.approx(1.0, abs=0.01) or norm == pytest.approx(0.0, abs=1e-6)
