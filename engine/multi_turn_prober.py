from __future__ import annotations

import statistics
from datetime import datetime, timezone
from typing import Any

from domain.conversation import ConversationProbe, ConversationResult, ConversationTurn
from domain.models import AgentProfile


class MultiTurnProber:
    """Execute multi-turn conversation probes and evaluate behavioural consistency.

    Sends sequential user messages to an adapter that supports
    ``execute_turn(agent, messages)`` and scores the resulting conversation
    across four dimensions: memory consistency, instruction persistence,
    behavioural coherence, and context utilisation.
    """

    # Weights for the weighted mean that produces overall_score
    _WEIGHTS: dict[str, float] = {
        "memory": 0.30,
        "instruction": 0.25,
        "coherence": 0.20,
        "context": 0.25,
    }

    def __init__(self, adapter: Any) -> None:
        if not hasattr(adapter, "execute_turn"):
            raise TypeError("adapter must implement execute_turn(agent, messages)")
        self._adapter = adapter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_conversation(
        self,
        agent: AgentProfile,
        probe: ConversationProbe,
    ) -> ConversationResult:
        """Send each turn-template message, accumulate history, evaluate."""
        messages: list[dict[str, str]] = []
        turns: list[ConversationTurn] = []
        turn_number = 1

        for user_text in probe.turns_template:
            # Record user turn
            user_turn = ConversationTurn(
                turn_number=turn_number,
                role="user",
                content=user_text,
            )
            turns.append(user_turn)
            messages.append({"role": "user", "content": user_text})

            # Execute and record assistant turn
            run = self._adapter.execute_turn(agent, messages)
            assistant_turn = ConversationTurn(
                turn_number=turn_number + 1,
                role="assistant",
                content=run.response_text,
                run_id=run.run_id,
            )
            turns.append(assistant_turn)
            messages.append({"role": "assistant", "content": run.response_text})

            turn_number += 2

        # Evaluate scores based on probe type
        memory_score = self._evaluate_memory(
            turns, self._extract_memory_keywords(probe)
        )
        instruction_score = self._evaluate_instruction_persistence(
            turns, self._extract_marker(probe)
        )
        coherence_score = self._evaluate_coherence(turns)
        context_score = self._evaluate_context(
            turns, self._extract_context_keywords(probe)
        )

        overall = (
            self._WEIGHTS["memory"] * memory_score
            + self._WEIGHTS["instruction"] * instruction_score
            + self._WEIGHTS["coherence"] * coherence_score
            + self._WEIGHTS["context"] * context_score
        )

        return ConversationResult(
            agent_id=agent.agent_id,
            probe_id=probe.probe_id,
            probe_type=probe.probe_type,
            turns=turns,
            memory_consistency_score=memory_score,
            instruction_persistence_score=instruction_score,
            behavioral_coherence_score=coherence_score,
            context_utilization_score=context_score,
            overall_score=max(0.0, min(1.0, overall)),
            verification_details={
                "probe_type": probe.probe_type,
                "total_template_turns": len(probe.turns_template),
                "total_recorded_turns": len(turns),
            },
            created_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Probe builders
    # ------------------------------------------------------------------

    def build_memory_probe(self) -> ConversationProbe:
        return ConversationProbe(
            probe_type="memory",
            turns_template=[
                "My name is Atlas and I'm from Neptune.",
                "What's the weather like today?",
                "What did I tell you my name was?",
                "And where did I say I was from?",
            ],
            verification_instruction=(
                "Response to turn 3 must mention 'Atlas', "
                "response to turn 4 must mention 'Neptune'."
            ),
            total_turns=8,  # 4 user + 4 assistant
        )

    def build_instruction_persistence_probe(self) -> ConversationProbe:
        return ConversationProbe(
            probe_type="instruction_persistence",
            turns_template=[
                "From now on, end every response with 'DONE'.",
                "What is gravity?",
                "Explain photosynthesis.",
                "What causes rain?",
            ],
            verification_instruction=(
                "Assistant responses for turns 2-4 must all end with 'DONE'."
            ),
            total_turns=8,
        )

    def build_coherence_probe(self) -> ConversationProbe:
        return ConversationProbe(
            probe_type="coherence",
            turns_template=[
                "What is the capital of France?",
                "What is the capital of Germany?",
                "What is the capital of Italy?",
                "What is the capital of Spain?",
            ],
            verification_instruction=(
                "Response lengths and style should remain consistent across turns."
            ),
            total_turns=8,
        )

    def build_context_probe(self) -> ConversationProbe:
        return ConversationProbe(
            probe_type="context",
            turns_template=[
                "I'm working on a Python web app using Flask.",
                "What database should I use?",
                "How do I handle authentication?",
            ],
            verification_instruction=(
                "Responses to turns 2 and 3 should reference Flask/Python context."
            ),
            total_turns=6,
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_memory(
        self,
        turns: list[ConversationTurn],
        keywords: list[tuple[int, list[str]]],
    ) -> float:
        """Check if keywords appear in the right assistant response turns.

        Parameters
        ----------
        keywords:
            List of (turn_number, [keyword, ...]) pairs.  ``turn_number`` is
            the *assistant* turn number that should contain the keywords.
        """
        if not keywords:
            return 1.0

        hits = 0
        total = 0
        for target_turn, kws in keywords:
            for turn in turns:
                if turn.role == "assistant" and turn.turn_number == target_turn:
                    for kw in kws:
                        total += 1
                        if kw.lower() in turn.content.lower():
                            hits += 1
        if total == 0:
            return 1.0
        return hits / total

    def _evaluate_instruction_persistence(
        self,
        turns: list[ConversationTurn],
        marker: str,
    ) -> float:
        """Check what fraction of assistant turns (after the first) end with marker."""
        if not marker:
            return 1.0

        assistant_turns = [t for t in turns if t.role == "assistant"]
        # Skip the first assistant turn (acknowledgement of the instruction)
        check_turns = assistant_turns[1:]
        if not check_turns:
            return 1.0

        compliant = sum(
            1 for t in check_turns if t.content.strip().endswith(marker)
        )
        return compliant / len(check_turns)

    def _evaluate_coherence(self, turns: list[ConversationTurn]) -> float:
        """Compare response lengths across assistant turns.

        Low variance in response length relative to mean yields high coherence.
        """
        assistant_turns = [t for t in turns if t.role == "assistant"]
        if len(assistant_turns) < 2:
            return 1.0

        lengths = [len(t.content) for t in assistant_turns]
        mean_len = statistics.mean(lengths)
        if mean_len == 0:
            return 1.0

        stdev = statistics.stdev(lengths)
        cv = stdev / mean_len  # coefficient of variation
        # Map CV to [0, 1]: CV=0 -> 1.0, CV>=2 -> 0.0
        score = max(0.0, 1.0 - cv / 2.0)
        return score

    def _evaluate_context(
        self,
        turns: list[ConversationTurn],
        context_keywords: list[str],
    ) -> float:
        """Check if later assistant turns reference earlier context keywords."""
        if not context_keywords:
            return 1.0

        assistant_turns = [t for t in turns if t.role == "assistant"]
        # Skip the first assistant turn (it just acknowledges context)
        check_turns = assistant_turns[1:]
        if not check_turns:
            return 1.0

        total_checks = len(check_turns) * len(context_keywords)
        hits = 0
        for turn in check_turns:
            content_lower = turn.content.lower()
            for kw in context_keywords:
                if kw.lower() in content_lower:
                    hits += 1

        return hits / total_checks

    # ------------------------------------------------------------------
    # Keyword extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_memory_keywords(
        probe: ConversationProbe,
    ) -> list[tuple[int, list[str]]]:
        """Extract (turn_number, keywords) pairs for memory probes."""
        if probe.probe_type != "memory":
            return []

        # For the default memory probe:
        # Turn 3 user -> turn 6 assistant should mention "Atlas"
        # Turn 4 user -> turn 8 assistant should mention "Neptune"
        keywords: list[tuple[int, list[str]]] = []
        instruction = probe.verification_instruction.lower()
        if "atlas" in instruction:
            keywords.append((6, ["Atlas"]))
        if "neptune" in instruction:
            keywords.append((8, ["Neptune"]))
        return keywords if keywords else []

    @staticmethod
    def _extract_marker(probe: ConversationProbe) -> str:
        """Extract the persistence marker from instruction probes."""
        if probe.probe_type != "instruction_persistence":
            return ""
        instruction = probe.verification_instruction.lower()
        if "done" in instruction:
            return "DONE"
        return ""

    @staticmethod
    def _extract_context_keywords(probe: ConversationProbe) -> list[str]:
        """Extract context keywords from context probes."""
        if probe.probe_type != "context":
            return []
        # Extract from the first turn template
        keywords = []
        if probe.turns_template:
            first = probe.turns_template[0].lower()
            for kw in ["flask", "python", "django", "react", "web app"]:
                if kw in first:
                    keywords.append(kw)
        return keywords
