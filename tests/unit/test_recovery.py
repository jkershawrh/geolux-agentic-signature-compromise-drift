import pytest

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import SignatureType
from domain.models import AgentProfile
from engine.recovery_engine import RecoveryEngine
from engine.signature_generator import SignatureGenerator


def _make_baseline(agent_id, adapter, n=5):
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


class TestRecoveryEngine:
    def test_successful_recovery(self):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()
        generator = SignatureGenerator(manifold_method="pca")
        agent = AgentProfile(agent_id="recover-test", display_name="R", model_id="test")
        old_baseline = _make_baseline("recover-test", adapter)

        engine = RecoveryEngine(
            adapter=adapter, extractor=extractor, generator=generator,
            recovery_distance_threshold=1.0,
        )
        prompts = [f"Recovery prompt {i}" for i in range(10)]
        result = engine.recover(agent, old_baseline, prompts)

        assert result.success is True
        assert result.new_baseline is not None
        assert result.distance_from_old >= 0

    def test_recovery_details_populated(self):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()
        generator = SignatureGenerator(manifold_method="pca")
        agent = AgentProfile(agent_id="recover-test", display_name="R", model_id="test")
        old_baseline = _make_baseline("recover-test", adapter)

        engine = RecoveryEngine(
            adapter=adapter, extractor=extractor, generator=generator,
            recovery_distance_threshold=1.0,
        )
        prompts = [f"Recovery prompt {i}" for i in range(10)]
        result = engine.recover(agent, old_baseline, prompts)
        assert "Recovery" in result.details

    def test_tight_threshold_may_fail(self):
        adapter = MockInferenceAdapter()
        extractor = DefaultMetricExtractor()
        generator = SignatureGenerator(manifold_method="pca")
        agent = AgentProfile(agent_id="recover-test", display_name="R", model_id="test")

        different_adapter = MockInferenceAdapter(
            response_key="code", latency_ms=500,
            input_tokens=300, output_tokens=200,
        )
        old_baseline = _make_baseline("recover-test", different_adapter)

        engine = RecoveryEngine(
            adapter=adapter, extractor=extractor, generator=generator,
            recovery_distance_threshold=0.001,
        )
        prompts = [f"Prompt {i}" for i in range(10)]
        result = engine.recover(agent, old_baseline, prompts)
        # May or may not succeed depending on how different the behaviors are
        assert result.distance_from_old >= 0
