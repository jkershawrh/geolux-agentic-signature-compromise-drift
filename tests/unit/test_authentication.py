import pytest

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import SignatureType
from domain.geometry import GeometricSignature
from domain.models import AgentProfile
from engine.authentication import AuthenticationEngine
from engine.signature_generator import SignatureGenerator


@pytest.fixture
def auth_engine():
    return AuthenticationEngine(distance_threshold=0.5, cosine_threshold=0.85)


@pytest.fixture
def generator():
    return SignatureGenerator(manifold_method="pca")


def _make_signature(agent_id, adapter, extractor, generator, num_runs=5):
    agent = AgentProfile(
        agent_id=agent_id,
        display_name=f"Agent {agent_id}",
        model_id="claude-sonnet-4-20250514",
    )
    metrics_per_run = []
    run_ids = []
    for i in range(num_runs):
        run = adapter.execute(agent, f"Prompt {i}")
        metrics = extractor.extract(run)
        metrics_per_run.append(metrics)
        run_ids.append(run.run_id)
    return generator.generate(
        agent_id, metrics_per_run, run_ids,
        signature_type=SignatureType.BASELINE,
    )


class TestAuthenticationEngine:
    def test_same_agent_authenticates(self, auth_engine, generator):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()

        baseline = _make_signature("agent-1", adapter, extractor, generator)
        current = _make_signature("agent-1", adapter, extractor, generator)

        result = auth_engine.verify(current, baseline)
        assert result.is_authentic is True
        assert result.confidence > 0.5
        assert result.geodesic_distance >= 0

    def test_different_agent_fails_authentication(self, auth_engine, generator):
        extractor = DefaultMetricExtractor()

        adapter_a = MockInferenceAdapter(
            response_key="default", latency_ms=150,
            input_tokens=100, output_tokens=50,
        )
        adapter_b = MockInferenceAdapter(
            response_key="code", latency_ms=500,
            input_tokens=300, output_tokens=200,
            thinking_tokens=100, include_tool_calls=True,
        )

        baseline = _make_signature("agent-a", adapter_a, extractor, generator)
        impostor = _make_signature("agent-b", adapter_b, extractor, generator)

        result = auth_engine.verify(impostor, baseline)
        assert result.is_authentic is False or result.confidence < 0.9

    def test_confidence_in_range(self, auth_engine, generator):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()

        baseline = _make_signature("agent-1", adapter, extractor, generator)
        current = _make_signature("agent-1", adapter, extractor, generator)

        result = auth_engine.verify(current, baseline)
        assert 0.0 <= result.confidence <= 1.0

    def test_euclidean_distance_used_for_authentication(self, auth_engine, generator):
        """Authentication decisions use Euclidean distance, not geodesic."""
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()

        baseline = _make_signature("agent-1", adapter, extractor, generator)
        current = _make_signature("agent-1", adapter, extractor, generator)

        result = auth_engine.verify(current, baseline)
        # The details should report euclidean_distance for the decision
        assert "euclidean_distance" in result.details
        # Geodesic distance is still reported (for drift detection) but not
        # used for the pass/fail decision
        assert "geodesic_distance" in result.details
        # Euclidean distance should be <= threshold when is_authentic is True
        if result.is_authentic:
            assert result.euclidean_distance <= auth_engine._distance_threshold

    def test_same_agent_same_adapter_is_authentic(self, auth_engine, generator):
        """Same agent with same adapter must authenticate (the core bug fix)."""
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()

        baseline = _make_signature("agent-1", adapter, extractor, generator)
        current = _make_signature("agent-1", adapter, extractor, generator)

        result = auth_engine.verify(current, baseline)
        assert result.is_authentic is True, (
            f"Same agent with same adapter should authenticate. "
            f"euclidean_distance={result.euclidean_distance:.4f}, "
            f"geodesic_distance={result.geodesic_distance:.4f}, "
            f"threshold={result.threshold_used}"
        )


class TestAgentIdentification:
    def test_identify_correct_agent(self, auth_engine, generator):
        extractor = DefaultMetricExtractor()

        adapter_a = MockInferenceAdapter(
            response_key="default", latency_ms=150,
            input_tokens=100, output_tokens=50,
        )
        adapter_b = MockInferenceAdapter(
            response_key="code", latency_ms=500,
            input_tokens=300, output_tokens=200,
            thinking_tokens=100, include_tool_calls=True,
        )

        baseline_a = _make_signature("agent-a", adapter_a, extractor, generator)
        baseline_b = _make_signature("agent-b", adapter_b, extractor, generator)

        current_a = _make_signature("agent-a", adapter_a, extractor, generator)

        agent_id, result = auth_engine.identify_agent(
            current_a, [baseline_a, baseline_b]
        )
        assert agent_id == "agent-a"
        assert result.is_authentic is True

    def test_no_candidates_returns_none(self, auth_engine):
        sig = GeometricSignature(
            agent_id="test",
            signature_type=SignatureType.SNAPSHOT,
            embedding_vector=[0.5] * 7,
            embedding_dimension=7,
            manifold_coordinates=[0.5, 0.5],
            metric_snapshot={"a": 0.5},
            run_ids=["r1"],
            num_runs=1,
            computation_method="pca",
        )
        agent_id, result = auth_engine.identify_agent(sig, [])
        assert agent_id is None
        assert result.is_authentic is False
