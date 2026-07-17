from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adapters.mock_adapter import MockConversationalAdapter
from domain.models import AgentProfile
from engine.multi_turn_prober import MultiTurnProber


def _make_prober_and_agent():
    adapter = MockConversationalAdapter()
    prober = MultiTurnProber(adapter)
    agent = AgentProfile(
        agent_id="prop-mt-agent",
        display_name="Property MT Agent",
        model_id="test-model",
    )
    return prober, agent


class TestMultiTurnProperties:
    @given(probe_type=st.sampled_from(["memory", "instruction_persistence", "coherence", "context"]))
    @settings(max_examples=10)
    def test_all_scores_in_unit_interval(self, probe_type):
        prober, agent = _make_prober_and_agent()
        builder = {
            "memory": prober.build_memory_probe,
            "instruction_persistence": prober.build_instruction_persistence_probe,
            "coherence": prober.build_coherence_probe,
            "context": prober.build_context_probe,
        }
        probe = builder[probe_type]()
        result = prober.execute_conversation(agent, probe)
        for attr in [
            "memory_consistency_score",
            "instruction_persistence_score",
            "behavioral_coherence_score",
            "context_utilization_score",
            "overall_score",
        ]:
            score = getattr(result, attr)
            assert 0.0 <= score <= 1.0, f"{attr} = {score} out of [0, 1]"

    @given(probe_type=st.sampled_from(["memory", "instruction_persistence", "coherence", "context"]))
    @settings(max_examples=10)
    def test_overall_score_bounded_by_min_max(self, probe_type):
        prober, agent = _make_prober_and_agent()
        builder = {
            "memory": prober.build_memory_probe,
            "instruction_persistence": prober.build_instruction_persistence_probe,
            "coherence": prober.build_coherence_probe,
            "context": prober.build_context_probe,
        }
        probe = builder[probe_type]()
        result = prober.execute_conversation(agent, probe)
        component_scores = [
            result.memory_consistency_score,
            result.instruction_persistence_score,
            result.behavioral_coherence_score,
            result.context_utilization_score,
        ]
        assert result.overall_score >= min(component_scores) * min(
            MultiTurnProber._WEIGHTS.values()
        ), "Overall score below weighted minimum"
        assert result.overall_score <= max(component_scores), (
            "Overall score above maximum component"
        )

    @given(probe_type=st.sampled_from(["memory", "instruction_persistence", "coherence", "context"]))
    @settings(max_examples=10)
    def test_turn_count_matches_template_times_two(self, probe_type):
        prober, agent = _make_prober_and_agent()
        builder = {
            "memory": prober.build_memory_probe,
            "instruction_persistence": prober.build_instruction_persistence_probe,
            "coherence": prober.build_coherence_probe,
            "context": prober.build_context_probe,
        }
        probe = builder[probe_type]()
        result = prober.execute_conversation(agent, probe)
        expected_turns = len(probe.turns_template) * 2  # user + assistant
        assert len(result.turns) == expected_turns, (
            f"Expected {expected_turns} turns, got {len(result.turns)}"
        )
