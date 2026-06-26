"""Contract tests verifying multi-turn prober satisfies expected interfaces."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adapters.mock_adapter import MockConversationalAdapter
from domain.conversation import ConversationResult
from domain.models import AgentProfile
from engine.multi_turn_prober import MultiTurnProber


def _make_result() -> ConversationResult:
    adapter = MockConversationalAdapter()
    prober = MultiTurnProber(adapter)
    agent = AgentProfile(
        agent_id="contract-mt-agent",
        display_name="Contract MT Agent",
        model_id="test-model",
    )
    probe = prober.build_memory_probe()
    return prober.execute_conversation(agent, probe)


class TestMultiTurnContracts:
    def test_output_is_conversation_result(self):
        result = _make_result()
        assert isinstance(result, ConversationResult)

    def test_conversational_adapter_protocol_satisfied(self):
        """MockConversationalAdapter satisfies the ConversationalAdapter protocol."""
        from adapters.interfaces import ConversationalAdapter

        adapter = MockConversationalAdapter()
        # Structural subtyping: check that execute_turn exists and is callable
        assert hasattr(adapter, "execute_turn")
        assert callable(adapter.execute_turn)

        # Verify it works with the expected signature
        agent = AgentProfile(
            agent_id="protocol-test",
            display_name="Protocol Test Agent",
            model_id="test-model",
        )
        messages = [{"role": "user", "content": "Hello"}]
        run = adapter.execute_turn(agent, messages)
        assert run.response_text  # non-empty response

    def test_turns_alternate_user_assistant(self):
        result = _make_result()
        assert len(result.turns) > 0
        for i, turn in enumerate(result.turns):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert turn.role == expected_role, (
                f"Turn {i} has role '{turn.role}', expected '{expected_role}'"
            )
