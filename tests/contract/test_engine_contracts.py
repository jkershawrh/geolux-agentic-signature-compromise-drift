"""Contract tests verifying engine components satisfy expected interfaces."""

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import DriftCategory, SignatureType
from domain.geometry import DriftMeasurement, GeometricSignature
from domain.models import AgentProfile
from engine.authentication import AuthenticationEngine, AuthenticationResult
from engine.compromise_detector import CompromiseDetector
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


class TestSignatureGeneratorContract:
    def test_output_is_geometric_signature(self):
        sig = _make_sig("test")
        assert isinstance(sig, GeometricSignature)

    def test_embedding_dimension_matches_vector(self):
        sig = _make_sig("test")
        assert sig.embedding_dimension == len(sig.embedding_vector)

    def test_stability_in_range(self):
        sig = _make_sig("test")
        assert sig.stability_score is None or 0.0 <= sig.stability_score <= 1.0

    def test_run_ids_match_num_runs(self):
        sig = _make_sig("test", n=7)
        assert sig.num_runs == 7
        assert len(sig.run_ids) == 7


class TestDriftDetectorContract:
    def test_output_is_drift_measurement(self):
        baseline = _make_sig("test")
        current = _make_sig("test")
        detector = DriftDetector()
        result = detector.detect(baseline, current)
        assert isinstance(result, DriftMeasurement)

    def test_distances_non_negative(self):
        baseline = _make_sig("test")
        current = _make_sig("test", {"response_key": "code", "latency_ms": 500})
        detector = DriftDetector()
        result = detector.detect(baseline, current)
        assert result.geodesic_distance >= 0
        assert result.euclidean_distance >= 0

    def test_magnitude_in_range(self):
        baseline = _make_sig("test")
        current = _make_sig("test")
        detector = DriftDetector()
        result = detector.detect(baseline, current)
        assert 0.0 <= result.drift_magnitude <= 1.0

    def test_cosine_in_range(self):
        baseline = _make_sig("test")
        current = _make_sig("test")
        detector = DriftDetector()
        result = detector.detect(baseline, current)
        assert -1.0 <= result.cosine_similarity <= 1.0

    def test_p_value_in_range(self):
        baseline = _make_sig("test")
        current = _make_sig("test")
        detector = DriftDetector()
        result = detector.detect(baseline, current)
        assert result.p_value is None or 0.0 <= result.p_value <= 1.0


class TestAuthenticationEngineContract:
    def test_output_is_authentication_result(self):
        sig = _make_sig("test")
        engine = AuthenticationEngine()
        result = engine.verify(sig, sig)
        assert isinstance(result, AuthenticationResult)

    def test_confidence_in_range(self):
        sig = _make_sig("test")
        engine = AuthenticationEngine()
        result = engine.verify(sig, sig)
        assert 0.0 <= result.confidence <= 1.0


class TestCompromiseDetectorContract:
    def test_no_alert_for_low_drift(self):
        detector = CompromiseDetector()
        drift = DriftMeasurement(
            agent_id="test",
            baseline_signature_id="s1",
            current_signature_id="s2",
            geodesic_distance=0.01,
            euclidean_distance=0.01,
            cosine_similarity=0.99,
            drift_category=DriftCategory.SEMANTIC,
            drift_magnitude=0.01,
            per_dimension_drift={},
            is_significant=False,
            p_value=0.9,
            compromise_probability=0.01,
        )
        assert detector.evaluate(drift) is None

    def test_alert_for_high_drift(self):
        detector = CompromiseDetector()
        drift = DriftMeasurement(
            agent_id="test",
            baseline_signature_id="s1",
            current_signature_id="s2",
            geodesic_distance=2.0,
            euclidean_distance=1.5,
            cosine_similarity=0.3,
            drift_category=DriftCategory.GOAL,
            drift_magnitude=0.95,
            per_dimension_drift={},
            is_significant=True,
            p_value=0.001,
            compromise_probability=0.95,
        )
        alert = detector.evaluate(drift)
        assert alert is not None
        assert alert.severity in ("warning", "critical")
