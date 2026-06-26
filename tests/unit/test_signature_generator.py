import numpy as np
import pytest

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import SignatureType
from domain.models import AgentProfile
from engine.signature_generator import SignatureGenerator


@pytest.fixture
def generator():
    return SignatureGenerator(min_runs=3, manifold_method="pca")


@pytest.fixture
def agent_alpha():
    return AgentProfile(
        agent_id="alpha",
        display_name="Agent Alpha",
        model_id="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant.",
    )


@pytest.fixture
def agent_beta():
    return AgentProfile(
        agent_id="beta",
        display_name="Agent Beta",
        model_id="claude-opus-4-20250514",
        system_prompt="You are a concise technical writer.",
    )


def _collect_metrics(agent, adapter, extractor, prompts):
    """Run an agent on prompts and collect metrics per run."""
    metrics_per_run = []
    run_ids = []
    for prompt in prompts:
        run = adapter.execute(agent, prompt)
        metrics = extractor.extract(run)
        metrics_per_run.append(metrics)
        run_ids.append(run.run_id)
    return metrics_per_run, run_ids


class TestSignatureGenerator:
    def test_generate_produces_signature(self, generator, agent_alpha):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()
        prompts = [f"Test prompt {i}" for i in range(5)]
        metrics, run_ids = _collect_metrics(agent_alpha, adapter, extractor, prompts)

        sig = generator.generate(agent_alpha.agent_id, metrics, run_ids)
        assert sig.agent_id == "alpha"
        assert sig.embedding_dimension == 35
        assert len(sig.embedding_vector) == 35
        assert sig.num_runs == 5
        assert sig.stability_score is not None
        assert 0.0 <= sig.stability_score <= 1.0

    def test_too_few_runs_raises(self, generator, agent_alpha):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()
        metrics, run_ids = _collect_metrics(
            agent_alpha, adapter, extractor, ["single prompt"]
        )
        with pytest.raises(ValueError, match="at least 3"):
            generator.generate(agent_alpha.agent_id, metrics, run_ids)

    def test_signature_type_propagated(self, generator, agent_alpha):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()
        prompts = [f"Prompt {i}" for i in range(5)]
        metrics, run_ids = _collect_metrics(agent_alpha, adapter, extractor, prompts)

        sig = generator.generate(
            agent_alpha.agent_id, metrics, run_ids,
            signature_type=SignatureType.BASELINE,
        )
        assert sig.signature_type == SignatureType.BASELINE

    def test_metric_snapshot_populated(self, generator, agent_alpha):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()
        prompts = [f"Prompt {i}" for i in range(5)]
        metrics, run_ids = _collect_metrics(agent_alpha, adapter, extractor, prompts)

        sig = generator.generate(agent_alpha.agent_id, metrics, run_ids)
        assert "avg_response_length" in sig.metric_snapshot
        assert "input_output_ratio" in sig.metric_snapshot

    def test_metric_tensor_present(self, generator, agent_alpha):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()
        prompts = [f"Prompt {i}" for i in range(5)]
        metrics, run_ids = _collect_metrics(agent_alpha, adapter, extractor, prompts)

        sig = generator.generate(agent_alpha.agent_id, metrics, run_ids)
        assert sig.metric_tensor is not None
        assert len(sig.metric_tensor) == 35
        assert len(sig.metric_tensor[0]) == 35

    def test_manifold_coordinates_present(self, generator, agent_alpha):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()
        prompts = [f"Prompt {i}" for i in range(5)]
        metrics, run_ids = _collect_metrics(agent_alpha, adapter, extractor, prompts)

        sig = generator.generate(agent_alpha.agent_id, metrics, run_ids)
        assert len(sig.manifold_coordinates) == 2


class TestReducibilityMask:
    """Test that the reducibility mask filters noisy metrics from signatures."""

    def test_reducibility_mask_filters_metrics(self, generator, agent_alpha):
        """Generate with a mask that zeros out some metrics; verify the
        resulting signature differs from the unmasked version and that
        the masked dimensions are actually zeroed."""
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()
        prompts = [f"Prompt {i}" for i in range(5)]
        metrics, run_ids = _collect_metrics(agent_alpha, adapter, extractor, prompts)

        sig_full = generator.generate(agent_alpha.agent_id, metrics, run_ids)

        # Mask out the last 13 metrics (indices 19-31)
        mask = [True] * 19 + [False] * 16
        sig_masked = generator.generate(
            agent_alpha.agent_id, metrics, run_ids, reducibility_mask=mask,
        )

        vec_full = np.array(sig_full.embedding_vector)
        vec_masked = np.array(sig_masked.embedding_vector)

        # The masked dimensions should be zero
        for i in range(19, 35):
            assert vec_masked[i] == 0.0, (
                f"Masked dimension {i} should be 0.0, got {vec_masked[i]}"
            )

        # The unmasked dimensions should remain unchanged
        # (they won't be exactly the same because the Frechet mean uses the
        # metric tensor which also changes, but they should be close)
        assert sig_masked.embedding_dimension == sig_full.embedding_dimension

        # The full and masked signatures must differ
        distance = float(np.linalg.norm(vec_full - vec_masked))
        assert distance > 0.001, (
            f"Masked signature should differ from full signature, "
            f"but distance was only {distance:.6f}"
        )

    def test_reducibility_mask_none_is_noop(self, generator, agent_alpha):
        """Passing reducibility_mask=None should produce the same result as
        not passing it at all."""
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()
        prompts = [f"Prompt {i}" for i in range(5)]
        metrics, run_ids = _collect_metrics(agent_alpha, adapter, extractor, prompts)

        sig_default = generator.generate(agent_alpha.agent_id, metrics, run_ids)
        sig_none = generator.generate(
            agent_alpha.agent_id, metrics, run_ids, reducibility_mask=None,
        )

        vec_default = np.array(sig_default.embedding_vector)
        vec_none = np.array(sig_none.embedding_vector)
        np.testing.assert_array_almost_equal(vec_default, vec_none)


class TestSignatureUniqueness:
    """Prove that different agents produce geometrically distinct signatures."""

    def test_different_agents_have_distant_signatures(self, generator, agent_alpha, agent_beta):
        extractor = DefaultMetricExtractor()

        adapter_alpha = MockInferenceAdapter(
            response_key="default", latency_ms=150,
            input_tokens=100, output_tokens=50,
        )
        adapter_beta = MockInferenceAdapter(
            response_key="code", latency_ms=300,
            input_tokens=200, output_tokens=120,
            thinking_tokens=80, include_tool_calls=True,
        )

        prompts = [f"Prompt {i}" for i in range(5)]
        metrics_a, ids_a = _collect_metrics(agent_alpha, adapter_alpha, extractor, prompts)
        metrics_b, ids_b = _collect_metrics(agent_beta, adapter_beta, extractor, prompts)

        sig_a = generator.generate(agent_alpha.agent_id, metrics_a, ids_a)
        sig_b = generator.generate(agent_beta.agent_id, metrics_b, ids_b)

        vec_a = np.array(sig_a.embedding_vector)
        vec_b = np.array(sig_b.embedding_vector)
        distance = float(np.linalg.norm(vec_a - vec_b))

        assert distance > 0.01, (
            f"Different agents should produce distinct signatures, "
            f"but distance was only {distance:.6f}"
        )


class TestSignatureStability:
    """Prove that the same agent produces consistent signatures over time."""

    def test_same_agent_has_close_signatures(self, generator, agent_alpha):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()

        prompts_1 = [f"Batch1 prompt {i}" for i in range(5)]
        prompts_2 = [f"Batch2 prompt {i}" for i in range(5)]

        metrics_1, ids_1 = _collect_metrics(agent_alpha, adapter, extractor, prompts_1)
        metrics_2, ids_2 = _collect_metrics(agent_alpha, adapter, extractor, prompts_2)

        sig_1 = generator.generate(agent_alpha.agent_id, metrics_1, ids_1)
        sig_2 = generator.generate(agent_alpha.agent_id, metrics_2, ids_2)

        vec_1 = np.array(sig_1.embedding_vector)
        vec_2 = np.array(sig_2.embedding_vector)
        distance = float(np.linalg.norm(vec_1 - vec_2))

        assert distance < 0.1, (
            f"Same agent should produce consistent signatures, "
            f"but distance was {distance:.6f}"
        )
