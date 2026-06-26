import pytest

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import SignatureType
from domain.models import AgentProfile
from engine.baseline_engine import BaselineEngine
from engine.signature_generator import SignatureGenerator


@pytest.fixture
def agent():
    return AgentProfile(
        agent_id="baseline-test-agent",
        display_name="Baseline Test Agent",
        model_id="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant.",
    )


@pytest.fixture
def prompts():
    return [f"Controlled prompt number {i}" for i in range(10)]


class TestBaselineEngine:
    def test_establish_baseline(self, agent, prompts):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()
        generator = SignatureGenerator(manifold_method="pca")
        engine = BaselineEngine(
            adapter=adapter, extractor=extractor, generator=generator,
            min_runs=5, convergence_epsilon=0.1, convergence_window=2,
        )

        result = engine.establish_baseline(agent, prompts)
        assert result.signature is not None
        assert result.signature.signature_type == SignatureType.BASELINE
        assert result.num_runs == 10
        assert len(result.runs) == 10
        assert len(result.all_metrics) == 10

    def test_convergence_detected(self, agent, prompts):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()
        generator = SignatureGenerator(manifold_method="pca")
        engine = BaselineEngine(
            adapter=adapter, extractor=extractor, generator=generator,
            min_runs=5, convergence_epsilon=1.0, convergence_window=2,
        )

        result = engine.establish_baseline(agent, prompts)
        assert result.is_converged is True

    def test_non_convergence_with_tight_epsilon(self, agent):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()
        generator = SignatureGenerator(min_runs=3, manifold_method="pca")
        engine = BaselineEngine(
            adapter=adapter, extractor=extractor, generator=generator,
            min_runs=3, convergence_epsilon=1e-20, convergence_window=5,
        )

        short_prompts = [f"Prompt {i}" for i in range(4)]
        result = engine.establish_baseline(agent, short_prompts)
        assert result.is_converged is False

    def test_baseline_signature_has_correct_run_count(self, agent, prompts):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()
        generator = SignatureGenerator(manifold_method="pca")
        engine = BaselineEngine(
            adapter=adapter, extractor=extractor, generator=generator,
        )

        result = engine.establish_baseline(agent, prompts)
        assert result.signature.num_runs == len(prompts)
        assert len(result.signature.run_ids) == len(prompts)

    def test_convergence_distances_tracked(self, agent, prompts):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()
        generator = SignatureGenerator(manifold_method="pca")
        engine = BaselineEngine(
            adapter=adapter, extractor=extractor, generator=generator,
        )

        result = engine.establish_baseline(agent, prompts)
        assert len(result.convergence_distances) > 0
        for d in result.convergence_distances:
            assert d >= 0
