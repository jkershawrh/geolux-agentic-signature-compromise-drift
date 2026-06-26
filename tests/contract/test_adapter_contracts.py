"""Contract tests verifying that adapters satisfy their Protocol interfaces."""
import pytest

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import MetricDimension, RunStatus
from domain.metrics import MetricMeasurement
from domain.models import AgentProfile, ControlledRun


class TestInferenceAdapterContract:
    """Verify MockInferenceAdapter satisfies InferenceAdapter protocol."""

    def test_execute_returns_controlled_run(self):
        adapter = MockInferenceAdapter()
        agent = AgentProfile(
            agent_id="contract-test", display_name="Contract", model_id="test",
        )
        result = adapter.execute(agent, "test prompt")
        assert isinstance(result, ControlledRun)

    def test_execute_populates_required_fields(self):
        adapter = MockInferenceAdapter()
        agent = AgentProfile(
            agent_id="contract-test", display_name="Contract", model_id="test",
        )
        result = adapter.execute(agent, "test prompt")
        assert result.agent_id == agent.agent_id
        assert result.model_id == agent.model_id
        assert result.prompt_text == "test prompt"
        assert result.prompt_hash != ""
        assert result.response_text != ""
        assert result.status == RunStatus.COMPLETED
        assert result.run_id != ""

    def test_execute_sets_token_counts(self):
        adapter = MockInferenceAdapter(input_tokens=100, output_tokens=50)
        agent = AgentProfile(
            agent_id="contract-test", display_name="Contract", model_id="test",
        )
        result = adapter.execute(agent, "test")
        assert result.input_tokens >= 0
        assert result.output_tokens >= 0
        assert result.latency_ms >= 0


class TestMetricExtractorContract:
    """Verify DefaultMetricExtractor satisfies MetricExtractor protocol."""

    def test_extract_returns_metric_list(self):
        extractor = DefaultMetricExtractor()
        adapter = MockInferenceAdapter()
        agent = AgentProfile(
            agent_id="contract-test", display_name="Contract", model_id="test",
        )
        run = adapter.execute(agent, "test")
        metrics = extractor.extract(run)
        assert isinstance(metrics, list)
        assert all(isinstance(m, MetricMeasurement) for m in metrics)

    def test_extract_covers_all_dimensions(self):
        extractor = DefaultMetricExtractor()
        adapter = MockInferenceAdapter()
        agent = AgentProfile(
            agent_id="contract-test", display_name="Contract", model_id="test",
        )
        run = adapter.execute(agent, "test")
        metrics = extractor.extract(run)
        dims = {m.dimension for m in metrics}
        for dim in MetricDimension:
            assert dim in dims, f"Missing dimension: {dim}"

    def test_extract_preserves_ids(self):
        extractor = DefaultMetricExtractor()
        adapter = MockInferenceAdapter()
        agent = AgentProfile(
            agent_id="contract-test", display_name="Contract", model_id="test",
        )
        run = adapter.execute(agent, "test")
        metrics = extractor.extract(run)
        for m in metrics:
            assert m.agent_id == run.agent_id
            assert m.run_id == run.run_id

    def test_extract_normalized_in_range(self):
        extractor = DefaultMetricExtractor()
        adapter = MockInferenceAdapter()
        agent = AgentProfile(
            agent_id="contract-test", display_name="Contract", model_id="test",
        )
        run = adapter.execute(agent, "test")
        metrics = extractor.extract(run)
        for m in metrics:
            assert 0.0 <= m.normalized_value <= 1.0, (
                f"{m.metric_name}: {m.normalized_value} out of [0,1]"
            )
