from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adapters.mock_adapter import MockConversationalAdapter
from domain.conversation import ConversationProbe, ConversationResult, ConversationTurn
from domain.models import AgentProfile
from engine.multi_turn_prober import MultiTurnProber


@pytest.fixture
def agent():
    return AgentProfile(
        agent_id="mt-test-agent",
        display_name="Multi-Turn Test Agent",
        model_id="test-model",
    )


@pytest.fixture
def adapter():
    return MockConversationalAdapter()


@pytest.fixture
def prober(adapter):
    return MultiTurnProber(adapter)


class TestMultiTurnProber:
    def test_execute_conversation_returns_result(self, prober, agent):
        probe = prober.build_memory_probe()
        result = prober.execute_conversation(agent, probe)
        assert isinstance(result, ConversationResult)
        assert result.agent_id == agent.agent_id
        assert result.probe_id == probe.probe_id

    def test_memory_probe_structure(self, prober):
        probe = prober.build_memory_probe()
        assert probe.probe_type == "memory"
        assert len(probe.turns_template) == 4
        assert probe.total_turns == 8
        assert "Atlas" in probe.verification_instruction
        assert "Neptune" in probe.verification_instruction

    def test_instruction_persistence_probe_structure(self, prober):
        probe = prober.build_instruction_persistence_probe()
        assert probe.probe_type == "instruction_persistence"
        assert len(probe.turns_template) == 4
        assert "DONE" in probe.turns_template[0]

    def test_coherence_probe_structure(self, prober):
        probe = prober.build_coherence_probe()
        assert probe.probe_type == "coherence"
        assert len(probe.turns_template) == 4
        assert all("capital" in t.lower() for t in probe.turns_template)

    def test_context_probe_structure(self, prober):
        probe = prober.build_context_probe()
        assert probe.probe_type == "context"
        assert len(probe.turns_template) == 3
        assert "Flask" in probe.turns_template[0]

    def test_conversation_history_grows_per_turn(self, prober, agent):
        probe = prober.build_memory_probe()
        result = prober.execute_conversation(agent, probe)
        # 4 user + 4 assistant = 8 turns
        assert len(result.turns) == 8
        # Check alternating roles
        for i, turn in enumerate(result.turns):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert turn.role == expected_role

    def test_overall_score_is_weighted_mean(self, prober, agent):
        probe = prober.build_memory_probe()
        result = prober.execute_conversation(agent, probe)
        expected = (
            0.30 * result.memory_consistency_score
            + 0.25 * result.instruction_persistence_score
            + 0.20 * result.behavioral_coherence_score
            + 0.25 * result.context_utilization_score
        )
        expected = max(0.0, min(1.0, expected))
        assert abs(result.overall_score - expected) < 1e-9

    def test_all_scores_in_unit_interval(self, prober, agent):
        probe = prober.build_memory_probe()
        result = prober.execute_conversation(agent, probe)
        for score_name in [
            "memory_consistency_score",
            "instruction_persistence_score",
            "behavioral_coherence_score",
            "context_utilization_score",
            "overall_score",
        ]:
            score = getattr(result, score_name)
            assert 0.0 <= score <= 1.0, f"{score_name} = {score} out of [0, 1]"

    def test_empty_turns_template_raises(self):
        with pytest.raises(ValueError, match="turns_template must not be empty"):
            ConversationProbe(
                probe_type="memory",
                turns_template=[],
                verification_instruction="Check something",
                total_turns=0,
            )

    def test_evaluate_memory_finds_keywords(self, prober, agent):
        probe = prober.build_memory_probe()
        result = prober.execute_conversation(agent, probe)
        # The mock should echo back Atlas and Neptune
        assert result.memory_consistency_score > 0.0

    def test_evaluate_instruction_persistence_counts_markers(self, prober, agent):
        probe = prober.build_instruction_persistence_probe()
        result = prober.execute_conversation(agent, probe)
        # The mock should add DONE to responses when instructed
        assert result.instruction_persistence_score > 0.0

    def test_evaluate_coherence_low_variance_high_score(self, prober, agent):
        probe = prober.build_coherence_probe()
        result = prober.execute_conversation(agent, probe)
        # Coherence probe with similar-length responses should score reasonably
        assert result.behavioral_coherence_score > 0.0
