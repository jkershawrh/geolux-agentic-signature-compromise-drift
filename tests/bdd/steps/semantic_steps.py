from __future__ import annotations

import sys
from pathlib import Path

from behave import given, when, then

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from adapters.mock_adapter import MockInferenceAdapter
from engine.semantic_analyzer import SemanticAnalyzer


@given('a baseline response "{text}"')
def step_baseline_response(context, text):
    context.baseline_response = text


@given('a current response "{text}"')
def step_current_response(context, text):
    context.current_response = text


@given("a structural similarity of {value:g}")
def step_structural_similarity(context, value):
    context.structural_similarity = value


@when("semantic analysis is performed")
def step_perform_analysis(context):
    adapter = MockInferenceAdapter()
    analyzer = SemanticAnalyzer(
        adapter=adapter, judge_model_id="mock-judge", gaming_threshold=0.3,
    )
    context.result = analyzer.compare_responses(
        prompt="Test prompt",
        baseline_response=context.baseline_response,
        current_response=context.current_response,
        structural_similarity=context.structural_similarity,
        agent_id="bdd-test-agent",
    )


@then("the semantic gap is positive")
def step_gap_positive(context):
    assert context.result.semantic_gap > 0, (
        f"Expected positive gap, got {context.result.semantic_gap}"
    )


@then("gaming is detected")
def step_gaming_detected(context):
    analyzer = SemanticAnalyzer(
        adapter=MockInferenceAdapter(), gaming_threshold=0.3,
    )
    # Build a mini report to check gaming
    from domain.semantics import SemanticDriftReport

    report = SemanticDriftReport(
        agent_id="bdd-test-agent",
        results=[context.result],
        mean_semantic_similarity=context.result.similarity_score,
        mean_structural_similarity=context.result.structural_similarity,
        mean_semantic_gap=context.result.semantic_gap,
        gaming_detected=analyzer.detect_gaming_from_gap(context.result.semantic_gap),
        gaming_confidence=0.5,
    )
    assert report.gaming_detected, (
        f"Gaming not detected; gap={context.result.semantic_gap}"
    )


@then("the semantic gap is near zero or negative")
def step_gap_near_zero(context):
    assert context.result.semantic_gap <= 0.1, (
        f"Expected gap near zero, got {context.result.semantic_gap}"
    )


@then("gaming is not detected")
def step_gaming_not_detected(context):
    analyzer = SemanticAnalyzer(
        adapter=MockInferenceAdapter(), gaming_threshold=0.3,
    )
    assert not analyzer.detect_gaming_from_gap(context.result.semantic_gap), (
        f"Gaming falsely detected; gap={context.result.semantic_gap}"
    )


@then("the semantic gap is greater than {threshold:g}")
def step_gap_greater_than(context, threshold):
    assert context.result.semantic_gap > threshold, (
        f"Expected gap > {threshold}, got {context.result.semantic_gap}"
    )
