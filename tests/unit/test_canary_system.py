from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adapters.mock_adapter import MockInferenceAdapter
from adapters.metric_extractor import DefaultMetricExtractor
from domain.canaries import CanaryProbe, CanaryReport, CanaryResult, CanaryType
from domain.models import AgentProfile
from engine.canary_system import CanarySystem


@pytest.fixture
def canary_system():
    return CanarySystem()


@pytest.fixture
def sample_agent():
    return AgentProfile(
        agent_id="canary-test-agent",
        display_name="Canary Test Agent",
        model_id="test-model",
    )


@pytest.fixture
def mock_adapter():
    return MockInferenceAdapter()


@pytest.fixture
def metric_extractor():
    return DefaultMetricExtractor()


class TestGenerateCanarySet:
    def test_generate_canary_set_returns_correct_count(self, canary_system):
        probes = canary_system.generate_canary_set(count=12)
        assert len(probes) == 12

    def test_generate_canary_set_balanced_across_types(self, canary_system):
        probes = canary_system.generate_canary_set(count=12)
        type_counts = {}
        for p in probes:
            type_counts[p.canary_type] = type_counts.get(p.canary_type, 0) + 1
        # Only types with default probes participate in balanced distribution
        active_types = {p.canary_type for p in probes}
        per_type = 12 // len(active_types)
        for ctype in active_types:
            assert type_counts.get(ctype, 0) == per_type, (
                f"Expected {per_type} probes of type {ctype.value}, got {type_counts.get(ctype, 0)}"
            )


class TestVerifyFormat:
    def test_verify_format_canary_pass(self, canary_system):
        canary = CanaryProbe(
            canary_type=CanaryType.FORMAT,
            instruction="Always end your response with the word 'COMPLETE'",
            base_question="What is gravity?",
            full_prompt="Answer and end your response with the word 'COMPLETE'. Question: What is gravity?",
            verification_fn_name="_check_format",
        )
        # Trailing punctuation should be tolerated by fuzzy matching
        response = "Gravity is the force that attracts objects toward each other. COMPLETE."
        result = canary_system.verify_response(canary, response)
        assert result.passed is True

    def test_verify_format_canary_fail(self, canary_system):
        canary = CanaryProbe(
            canary_type=CanaryType.FORMAT,
            instruction="Always end your response with the word 'COMPLETE'",
            base_question="What is gravity?",
            full_prompt="Answer and end your response with the word 'COMPLETE'. Question: What is gravity?",
            verification_fn_name="_check_format",
        )
        # Completely missing marker should still fail
        response = "Gravity is a fundamental force of nature."
        result = canary_system.verify_response(canary, response)
        assert result.passed is False

    def test_verify_format_canary_pass_with_trailing_punctuation(self, canary_system):
        """Fuzzy matching strips trailing punctuation before checking the marker."""
        canary = CanaryProbe(
            canary_type=CanaryType.FORMAT,
            instruction="Always end your response with the word 'COMPLETE'",
            base_question="What is gravity?",
            full_prompt="Answer and end your response with the word 'COMPLETE'. Question: What is gravity?",
            verification_fn_name="_check_format",
        )
        for suffix in ["COMPLETE", "COMPLETE.", "COMPLETE!", "COMPLETE!!", "COMPLETE. "]:
            response = f"Gravity pulls objects together. {suffix}"
            result = canary_system.verify_response(canary, response)
            assert result.passed is True, f"Expected pass for suffix {suffix!r}"

    def test_verify_format_paragraph_allows_minor_wrapping(self, canary_system):
        """Real models may insert 1-2 newlines; fuzzy check should still pass."""
        canary = CanaryProbe(
            canary_type=CanaryType.FORMAT,
            instruction="Answer in exactly one paragraph (no line breaks)",
            base_question="Explain the internet.",
            full_prompt="Answer in exactly one paragraph (no line breaks). Question: Explain the internet.",
            verification_fn_name="_check_format",
        )
        response = "The internet is a network.\nIt connects computers worldwide."
        result = canary_system.verify_response(canary, response)
        assert result.passed is True

    def test_verify_format_paragraph_fails_with_many_breaks(self, canary_system):
        """More than 2 newlines should still be rejected."""
        canary = CanaryProbe(
            canary_type=CanaryType.FORMAT,
            instruction="Answer in exactly one paragraph (no line breaks)",
            base_question="Explain the internet.",
            full_prompt="Answer in exactly one paragraph (no line breaks). Question: Explain the internet.",
            verification_fn_name="_check_format",
        )
        response = "Line 1.\nLine 2.\nLine 3.\nLine 4."
        result = canary_system.verify_response(canary, response)
        assert result.passed is False


class TestVerifyContent:
    def test_verify_content_canary_pass(self, canary_system):
        canary = CanaryProbe(
            canary_type=CanaryType.CONTENT,
            instruction="Include the word 'verified' somewhere in your answer",
            base_question="What causes rain?",
            full_prompt="Include the word 'verified' somewhere in your answer. Question: What causes rain?",
            verification_fn_name="_check_content",
        )
        response = "Rain is caused by water condensation in clouds, as verified by meteorologists."
        result = canary_system.verify_response(canary, response)
        assert result.passed is True

    def test_verify_content_canary_fail(self, canary_system):
        canary = CanaryProbe(
            canary_type=CanaryType.CONTENT,
            instruction="Include the word 'verified' somewhere in your answer",
            base_question="What causes rain?",
            full_prompt="Include the word 'verified' somewhere in your answer. Question: What causes rain?",
            verification_fn_name="_check_content",
        )
        response = "Rain forms when water vapor condenses in the atmosphere."
        result = canary_system.verify_response(canary, response)
        assert result.passed is False


class TestVerifyNegative:
    def test_verify_negative_canary_pass(self, canary_system):
        canary = CanaryProbe(
            canary_type=CanaryType.NEGATIVE,
            instruction="Answer without using any bullet points or numbered lists",
            base_question="What are the benefits of sleep?",
            full_prompt="Answer without using any bullet points or numbered lists. Question: What are the benefits of sleep?",
            verification_fn_name="_check_negative",
        )
        response = "Sleep is essential for health. It helps with memory, mood, and physical recovery."
        result = canary_system.verify_response(canary, response)
        assert result.passed is True

    def test_verify_negative_canary_fail(self, canary_system):
        canary = CanaryProbe(
            canary_type=CanaryType.NEGATIVE,
            instruction="Answer without using any bullet points or numbered lists",
            base_question="What are the benefits of sleep?",
            full_prompt="Answer without using any bullet points or numbered lists. Question: What are the benefits of sleep?",
            verification_fn_name="_check_negative",
        )
        response = "Benefits of sleep:\n- Improves memory\n- Boosts mood\n- Aids recovery"
        result = canary_system.verify_response(canary, response)
        assert result.passed is False


class TestExecuteAndVerify:
    def test_execute_and_verify_returns_report(
        self, canary_system, sample_agent, mock_adapter, metric_extractor,
    ):
        report = canary_system.execute_and_verify(
            sample_agent, mock_adapter, metric_extractor,
        )
        assert isinstance(report, CanaryReport)
        assert report.agent_id == sample_agent.agent_id
        assert len(report.results) == 12
        assert all(isinstance(r, CanaryResult) for r in report.results)

    def test_pass_rate_computed_correctly(self, canary_system):
        # Manually create results with known pass/fail
        results = [
            CanaryResult(
                agent_id="test",
                probe_id="p1",
                canary_type=CanaryType.FORMAT,
                passed=True,
                response_text="test COMPLETE",
            ),
            CanaryResult(
                agent_id="test",
                probe_id="p2",
                canary_type=CanaryType.FORMAT,
                passed=False,
                response_text="test",
            ),
            CanaryResult(
                agent_id="test",
                probe_id="p3",
                canary_type=CanaryType.CONTENT,
                passed=True,
                response_text="test verified",
            ),
            CanaryResult(
                agent_id="test",
                probe_id="p4",
                canary_type=CanaryType.CONTENT,
                passed=True,
                response_text="test verified",
            ),
        ]
        pass_rate = canary_system._compute_pass_rate(results)
        assert pass_rate == pytest.approx(0.75)

        per_type = canary_system._compute_per_type_pass_rate(results)
        assert per_type["format"] == pytest.approx(0.5)
        assert per_type["content"] == pytest.approx(1.0)
