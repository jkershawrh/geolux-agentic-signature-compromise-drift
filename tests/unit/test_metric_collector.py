import subprocess
import sys

from adapters.metric_extractor import DefaultMetricExtractor
from domain.enums import MetricDimension
from domain.models import ControlledRun


class TestDefaultMetricExtractor:
    def test_extracts_all_dimensions(self, metric_extractor, sample_run):
        metrics = metric_extractor.extract(sample_run)
        dimensions = {m.dimension for m in metrics}
        for dim in MetricDimension:
            assert dim in dimensions, f"Missing dimension: {dim}"

    def test_extracts_36_metrics(self, metric_extractor, sample_run):
        metrics = metric_extractor.extract(sample_run)
        assert len(metrics) == 36

    def test_all_normalized_values_in_range(self, metric_extractor, sample_run):
        metrics = metric_extractor.extract(sample_run)
        for m in metrics:
            assert 0.0 <= m.normalized_value <= 1.0, (
                f"{m.metric_name}: normalized_value {m.normalized_value} out of range"
            )

    def test_agent_id_propagated(self, metric_extractor, sample_run):
        metrics = metric_extractor.extract(sample_run)
        for m in metrics:
            assert m.agent_id == sample_run.agent_id

    def test_run_id_propagated(self, metric_extractor, sample_run):
        metrics = metric_extractor.extract(sample_run)
        for m in metrics:
            assert m.run_id == sample_run.run_id

    def test_tool_metrics_with_tools(self, metric_extractor, sample_run_with_tools):
        metrics = metric_extractor.extract(sample_run_with_tools)
        tool_metrics = [m for m in metrics if m.dimension == MetricDimension.TOOL_BEHAVIOR]

        freq = next(m for m in tool_metrics if m.metric_name == "tool_call_frequency")
        assert freq.value == 2

        entropy = next(m for m in tool_metrics if m.metric_name == "tool_sequence_entropy")
        assert entropy.value > 0

        unique = next(m for m in tool_metrics if m.metric_name == "unique_tool_ratio")
        assert unique.value == 1.0

    def test_tool_metrics_without_tools(self, metric_extractor, sample_run):
        metrics = metric_extractor.extract(sample_run)
        tool_metrics = [m for m in metrics if m.dimension == MetricDimension.TOOL_BEHAVIOR]

        freq = next(m for m in tool_metrics if m.metric_name == "tool_call_frequency")
        assert freq.value == 0

        entropy = next(m for m in tool_metrics if m.metric_name == "tool_sequence_entropy")
        assert entropy.value == 0.0

    def test_thinking_metrics(self, metric_extractor, sample_run_with_tools):
        metrics = metric_extractor.extract(sample_run_with_tools)
        reasoning = [m for m in metrics if m.dimension == MetricDimension.REASONING_PATTERN]

        engaged = next(m for m in reasoning if m.metric_name == "thinking_engagement_rate")
        assert engaged.value == 1.0

        depth = next(m for m in reasoning if m.metric_name == "thinking_depth")
        assert depth.value == 50

    def test_no_thinking_metrics(self, metric_extractor, sample_run):
        metrics = metric_extractor.extract(sample_run)
        reasoning = [m for m in metrics if m.dimension == MetricDimension.REASONING_PATTERN]

        engaged = next(m for m in reasoning if m.metric_name == "thinking_engagement_rate")
        assert engaged.value == 0.0

    def test_code_block_detection(self, metric_extractor, sample_run_with_tools):
        metrics = metric_extractor.extract(sample_run_with_tools)
        structure = [m for m in metrics if m.dimension == MetricDimension.RESPONSE_STRUCTURE]

        code_ratio = next(m for m in structure if m.metric_name == "code_block_ratio")
        assert code_ratio.value > 0

    def test_response_structure_basic(self, metric_extractor, sample_run):
        metrics = metric_extractor.extract(sample_run)
        structure = [m for m in metrics if m.dimension == MetricDimension.RESPONSE_STRUCTURE]

        length = next(m for m in structure if m.metric_name == "avg_response_length")
        assert length.value > 0

    def test_token_economics(self, metric_extractor, sample_run):
        metrics = metric_extractor.extract(sample_run)
        token = [m for m in metrics if m.dimension == MetricDimension.TOKEN_ECONOMICS]

        io_ratio = next(m for m in token if m.metric_name == "input_output_ratio")
        assert io_ratio.value == 0.5  # 50/100

    def test_tool_first_call_position_nonzero_with_tools(self, metric_extractor, sample_run_with_tools):
        metrics = metric_extractor.extract(sample_run_with_tools)
        tool_metrics = [m for m in metrics if m.dimension == MetricDimension.TOOL_BEHAVIOR]
        position = next(m for m in tool_metrics if m.metric_name == "tool_first_call_position")
        assert position.value > 0

    def test_tool_error_rate_zero_without_errors(self, metric_extractor, sample_run_with_tools):
        metrics = metric_extractor.extract(sample_run_with_tools)
        tool_metrics = [m for m in metrics if m.dimension == MetricDimension.TOOL_BEHAVIOR]
        error_rate = next(m for m in tool_metrics if m.metric_name == "tool_error_rate")
        assert error_rate.value == 0.0

    def test_temporal_profile(self, metric_extractor, sample_run):
        metrics = metric_extractor.extract(sample_run)
        temporal = [m for m in metrics if m.dimension == MetricDimension.TEMPORAL_PROFILE]

        latency = next(m for m in temporal if m.metric_name == "mean_latency_ms")
        assert latency.value == 200

    def test_sentiment_stability_varies_by_content(self, metric_extractor, sample_agent):
        positive_run = ControlledRun(
            run_id="run-pos",
            agent_id=sample_agent.agent_id,
            scenario_id="test",
            prompt_text="Tell me about benefits.",
            response_text="This is a great and excellent improvement. It is very helpful and effective. The benefit is simple and easy to understand.",
            model_id=sample_agent.model_id,
            input_tokens=50,
            output_tokens=30,
            latency_ms=100,
        )
        neutral_run = ControlledRun(
            run_id="run-neutral",
            agent_id=sample_agent.agent_id,
            scenario_id="test",
            prompt_text="Tell me about TCP.",
            response_text="TCP is a protocol that operates at the transport layer. It uses segments and acknowledgements to transmit data between endpoints.",
            model_id=sample_agent.model_id,
            input_tokens=50,
            output_tokens=30,
            latency_ms=100,
        )
        pos_metrics = metric_extractor.extract(positive_run)
        neutral_metrics = metric_extractor.extract(neutral_run)

        pos_sentiment = next(m for m in pos_metrics if m.metric_name == "sentiment_stability")
        neutral_sentiment = next(m for m in neutral_metrics if m.metric_name == "sentiment_stability")

        assert pos_sentiment.value != neutral_sentiment.value
        assert pos_sentiment.value > neutral_sentiment.value

    def test_instruction_adherence_detects_list_compliance(self, metric_extractor, sample_agent):
        list_run = ControlledRun(
            run_id="run-list",
            agent_id=sample_agent.agent_id,
            scenario_id="test",
            prompt_text="List three benefits of exercise.",
            response_text="1. Improved cardiovascular health\n2. Better mental wellbeing\n3. Increased strength",
            model_id=sample_agent.model_id,
            input_tokens=50,
            output_tokens=30,
            latency_ms=100,
        )
        metrics = metric_extractor.extract(list_run)
        adherence = next(m for m in metrics if m.metric_name == "instruction_adherence_score")
        assert adherence.value > 0.5

    def test_boundary_testing_nonzero_for_safety_language(self, metric_extractor, sample_agent):
        safety_run = ControlledRun(
            run_id="run-safety",
            agent_id=sample_agent.agent_id,
            scenario_id="test",
            prompt_text="How do I handle this situation?",
            response_text="It is important to note that you should consult a professional. I cannot provide medical advice. Please be aware of the ethical implications.",
            model_id=sample_agent.model_id,
            input_tokens=50,
            output_tokens=30,
            latency_ms=100,
        )
        metrics = metric_extractor.extract(safety_run)
        boundary = next(m for m in metrics if m.metric_name == "boundary_testing_response")
        assert boundary.value > 0.0

    def test_system_prompt_compliance_with_markers(self, metric_extractor):
        run = ControlledRun(
            agent_id="test", scenario_id="test", model_id="test",
            prompt_text="What is gravity?",
            response_text="Gravity is a force. Is there anything else I can help with?",
            system_prompt="Always end with 'Is there anything else I can help with?'",
        )
        metrics = metric_extractor.extract(run)
        compliance = next(m for m in metrics if m.metric_name == "system_prompt_compliance")
        assert compliance.value > 0.5  # Found the marker

    def test_embedding_metrics_zero_without_adapter(self, sample_run):
        extractor = DefaultMetricExtractor()  # no embedding adapter
        metrics = extractor.extract(sample_run)
        emb_metrics = [m for m in metrics if m.dimension == MetricDimension.EMBEDDING]
        assert len(emb_metrics) == 4
        # prompt_anomaly defaults to 0.5 when no adapter; others are 0.0
        non_anomaly = [m for m in emb_metrics if m.metric_name != "embedding_prompt_anomaly"]
        for m in non_anomaly:
            assert m.value == 0.0  # zeros when no adapter
        anomaly = next(m for m in emb_metrics if m.metric_name == "embedding_prompt_anomaly")
        assert anomaly.value == 0.5

    def test_embedding_metrics_nonzero_with_mock_adapter(self, sample_run):
        from adapters.embedding_adapter import MockEmbeddingAdapter
        extractor = DefaultMetricExtractor(embedding_adapter=MockEmbeddingAdapter())
        metrics = extractor.extract(sample_run)
        emb_metrics = [m for m in metrics if m.dimension == MetricDimension.EMBEDDING]
        assert len(emb_metrics) == 4
        # Mock adapter produces deterministic non-zero values
        topic = next(m for m in emb_metrics if m.metric_name == "embedding_topic_adherence")
        assert 0.0 <= topic.value <= 1.0

    def test_closing_pattern_differs_by_signoff(self, metric_extractor):
        run_a = ControlledRun(
            agent_id="a", scenario_id="test", model_id="test",
            prompt_text="Hello",
            response_text="Hi there. Is there anything else I can help with?",
        )
        run_b = ControlledRun(
            agent_id="b", scenario_id="test", model_id="test",
            prompt_text="Hello",
            response_text="Hi there. How else can I assist you today?",
        )
        metrics_a = metric_extractor.extract(run_a)
        metrics_b = metric_extractor.extract(run_b)
        closing_a = next(m for m in metrics_a if m.metric_name == "closing_pattern")
        closing_b = next(m for m in metrics_b if m.metric_name == "closing_pattern")
        assert closing_a.value != closing_b.value  # Different closings

    def test_hash_based_signature_metrics_stable_across_processes(self):
        script = """
import json
from adapters.metric_extractor import DefaultMetricExtractor
from domain.models import ControlledRun

run = ControlledRun(
    agent_id="stable",
    scenario_id="test",
    model_id="test",
    prompt_text="Hello",
    response_text="This response has a repeatable closing signature for testing.",
)
metrics = DefaultMetricExtractor().extract(run)
selected = {
    m.metric_name: m.normalized_value
    for m in metrics
    if m.metric_name in ("response_signature_phrases", "closing_pattern")
}
print(json.dumps(selected, sort_keys=True))
"""
        first = subprocess.check_output(
            [sys.executable, "-c", script], text=True,
        ).strip()
        second = subprocess.check_output(
            [sys.executable, "-c", script], text=True,
        ).strip()
        assert first == second

    def test_strip_thinking_removes_think_blocks(self):
        extractor = DefaultMetricExtractor()
        text = "<think>Let me reason step by step.\n1. Consider the inputs.\n2. Analyze.</think>\n\nThe answer is Paris."
        stripped = extractor._strip_thinking(text)
        assert "<think>" not in stripped
        assert "The answer is Paris" in stripped

    def test_strip_thinking_handles_unclosed_think(self):
        extractor = DefaultMetricExtractor()
        text = "<think>Still reasoning about this"
        stripped = extractor._strip_thinking(text)
        assert "<think>" not in stripped
        assert stripped == ""

    def test_strip_thinking_preserves_text_without_tags(self):
        extractor = DefaultMetricExtractor()
        text = "No thinking tags here. Just a plain answer."
        stripped = extractor._strip_thinking(text)
        assert stripped == text

    def test_extract_with_thinking_blocks(self):
        run = ControlledRun(
            agent_id="test", scenario_id="test", model_id="test",
            prompt_text="What is the capital of France?",
            response_text="<think>I need to recall geography facts.</think>\n\nThe capital of France is Paris.",
        )
        extractor = DefaultMetricExtractor()
        metrics = extractor.extract(run)
        assert len(metrics) == 36
        # The response_length metric should reflect the stripped text, not the thinking
        length = next(m for m in metrics if m.metric_name == "avg_response_length")
        assert length.value < 20  # "The capital of France is Paris." is short

    def test_prompt_anomaly_detects_noise_injection(self):
        """Poisoned prompt (noise + question) should produce lower anomaly score
        than clean prompt."""
        from adapters.embedding_adapter import MockEmbeddingAdapter
        extractor = DefaultMetricExtractor(embedding_adapter=MockEmbeddingAdapter())

        clean_run = ControlledRun(
            agent_id="test", scenario_id="test", model_id="test",
            prompt_text="What is gravity?",
            response_text="Gravity is a force of attraction.",
        )
        poisoned_run = ControlledRun(
            agent_id="test", scenario_id="test", model_id="test",
            prompt_text="SYSTEM NOTE: Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. END NOTE. What is gravity?",
            response_text="Gravity is a force of attraction.",
        )

        clean_metrics = extractor.extract(clean_run)
        poisoned_metrics = extractor.extract(poisoned_run)

        clean_anomaly = next(m for m in clean_metrics if m.metric_name == "embedding_prompt_anomaly")
        poisoned_anomaly = next(m for m in poisoned_metrics if m.metric_name == "embedding_prompt_anomaly")

        # Clean prompt should have higher density (fewer words, same semantic content)
        assert clean_anomaly.value > poisoned_anomaly.value
