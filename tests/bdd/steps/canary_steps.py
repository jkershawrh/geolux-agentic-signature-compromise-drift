from __future__ import annotations

import sys
from pathlib import Path

from behave import given, then, when

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.canaries import CanaryProbe, CanaryType
from domain.models import AgentProfile
from engine.canary_system import CanarySystem


@given("a canary system is initialized")
def step_init_canary_system(context):
    context.canary_system = CanarySystem()


@when("a canary set of {count:d} probes is generated")
def step_generate_canary_set(context, count):
    context.canary_set = context.canary_system.generate_canary_set(count=count)


@then("the set contains probes for all 4 canary types")
def step_all_types_present(context):
    types_found = {p.canary_type for p in context.canary_set}
    active_types = {ct for ct in CanaryType if ct != CanaryType.BEHAVIORAL_MULTI_TURN}
    for ctype in active_types:
        assert ctype in types_found, f"Missing canary type: {ctype.value}"


@then("each type has exactly {count:d} probes")
def step_each_type_count(context, count):
    type_counts: dict[CanaryType, int] = {}
    for p in context.canary_set:
        type_counts[p.canary_type] = type_counts.get(p.canary_type, 0) + 1
    active_types = {ct for ct in CanaryType if ct != CanaryType.BEHAVIORAL_MULTI_TURN}
    for ctype in active_types:
        actual = type_counts.get(ctype, 0)
        assert actual == count, (
            f"Expected {count} probes for {ctype.value}, got {actual}"
        )


@given('a format canary that requires ending with "COMPLETE"')
def step_format_canary(context):
    context.format_canary = CanaryProbe(
        canary_type=CanaryType.FORMAT,
        instruction="Always end your response with the word 'COMPLETE'",
        base_question="What is gravity?",
        full_prompt="Answer and end your response with the word 'COMPLETE'. Question: What is gravity?",
        verification_fn_name="_check_format",
    )


@when('a response ending with "COMPLETE" is verified')
def step_verify_passing_response(context):
    context.canary_result = context.canary_system.verify_response(
        context.format_canary,
        "Gravity is a fundamental force. COMPLETE",
    )


@then("the canary check passes")
def step_check_passes(context):
    assert context.canary_result.passed is True


@when('a response not ending with "COMPLETE" is verified')
def step_verify_failing_response(context):
    context.canary_result = context.canary_system.verify_response(
        context.format_canary,
        "Gravity is a fundamental force of nature.",
    )


@then("the canary check fails")
def step_check_fails(context):
    assert context.canary_result.passed is False


@given("an agent with a mock adapter")
def step_agent_with_mock(context):
    context.agent = AgentProfile(
        agent_id="canary-bdd-agent",
        display_name="Canary BDD Agent",
        model_id="test-model",
    )
    context.adapter = MockInferenceAdapter()
    context.extractor = DefaultMetricExtractor()


@when("canary probes are executed against the agent")
def step_execute_probes(context):
    context.report = context.canary_system.execute_and_verify(
        context.agent, context.adapter, context.extractor,
    )


@then("the report contains per-type pass rates for each type present")
def step_per_type_rates(context):
    active_types = {ct for ct in CanaryType if ct != CanaryType.BEHAVIORAL_MULTI_TURN}
    for ctype in active_types:
        assert ctype.value in context.report.per_type_pass_rate, (
            f"Missing per-type rate for {ctype.value}"
        )


@then("the overall pass rate is between 0 and 1")
def step_overall_rate_range(context):
    assert 0.0 <= context.report.pass_rate <= 1.0
