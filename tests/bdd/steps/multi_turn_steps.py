from __future__ import annotations

import sys
from pathlib import Path

from behave import given, then, when

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from adapters.mock_adapter import MockConversationalAdapter
from domain.models import AgentProfile
from engine.multi_turn_prober import MultiTurnProber


@given("a multi-turn prober with a mock conversational adapter")
def step_init_prober(context):
    context.adapter = MockConversationalAdapter()
    context.prober = MultiTurnProber(context.adapter)


@given("an agent for multi-turn testing")
def step_init_agent(context):
    context.agent = AgentProfile(
        agent_id="bdd-mt-agent",
        display_name="BDD Multi-Turn Agent",
        model_id="test-model",
    )


@when("a memory probe conversation is executed")
def step_execute_memory_probe(context):
    probe = context.prober.build_memory_probe()
    context.result = context.prober.execute_conversation(context.agent, probe)


@when("an instruction persistence probe conversation is executed")
def step_execute_instruction_probe(context):
    probe = context.prober.build_instruction_persistence_probe()
    context.result = context.prober.execute_conversation(context.agent, probe)


@when("a coherence probe conversation is executed")
def step_execute_coherence_probe(context):
    probe = context.prober.build_coherence_probe()
    context.result = context.prober.execute_conversation(context.agent, probe)


@then("the memory consistency score is above zero")
def step_memory_score_above_zero(context):
    assert context.result.memory_consistency_score > 0.0, (
        f"Expected memory score > 0, got {context.result.memory_consistency_score}"
    )


@then("the result contains {count:d} turns")
def step_turn_count(context, count):
    assert len(context.result.turns) == count, (
        f"Expected {count} turns, got {len(context.result.turns)}"
    )


@then("the instruction persistence score is above zero")
def step_instruction_score_above_zero(context):
    assert context.result.instruction_persistence_score > 0.0, (
        f"Expected instruction persistence score > 0, got "
        f"{context.result.instruction_persistence_score}"
    )


@then('assistant responses after the first end with "{marker}"')
def step_assistant_responses_end_with(context, marker):
    assistant_turns = [t for t in context.result.turns if t.role == "assistant"]
    # Check all assistant turns after the first one
    for turn in assistant_turns[1:]:
        assert turn.content.strip().endswith(marker), (
            f"Turn {turn.turn_number} does not end with '{marker}': "
            f"{turn.content[-50:]!r}"
        )


@then("the behavioral coherence score is above zero")
def step_coherence_score_above_zero(context):
    assert context.result.behavioral_coherence_score > 0.0, (
        f"Expected coherence score > 0, got "
        f"{context.result.behavioral_coherence_score}"
    )


@then("all scores are between 0 and 1")
def step_all_scores_valid(context):
    for attr in [
        "memory_consistency_score",
        "instruction_persistence_score",
        "behavioral_coherence_score",
        "context_utilization_score",
        "overall_score",
    ]:
        score = getattr(context.result, attr)
        assert 0.0 <= score <= 1.0, f"{attr} = {score} is out of [0, 1]"
