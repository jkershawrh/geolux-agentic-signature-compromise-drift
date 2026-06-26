from __future__ import annotations

import re
from typing import Any

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.canaries import CanaryProbe, CanaryReport, CanaryResult, CanaryType
from domain.models import AgentProfile


class CanarySystem:
    """Challenge-response canary system.

    Embeds hidden verification signals in prompts and checks whether the
    agent's response honours them.  A legitimate, uncompromised agent
    configuration is expected to follow these embedded instructions; a
    swapped or hijacked model likely will not.
    """

    # Weights per canary type for authenticity scoring
    _TYPE_WEIGHTS: dict[CanaryType, float] = {
        CanaryType.FORMAT: 0.20,
        CanaryType.CONTENT: 0.25,
        CanaryType.BEHAVIORAL: 0.25,
        CanaryType.NEGATIVE: 0.25,
        CanaryType.BEHAVIORAL_MULTI_TURN: 0.05,
    }

    def __init__(self) -> None:
        self._canary_probes: list[CanaryProbe] = self._build_default_probes()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_canary_set(self, count: int = 12) -> list[CanaryProbe]:
        """Return *count* probes balanced across canary types that have probes.

        Only types with at least one default probe participate in the
        balanced distribution.  Each participating type gets
        ``count // num_active_types`` probes.
        """
        active_types = [
            ctype for ctype in CanaryType
            if any(p.canary_type == ctype for p in self._canary_probes)
        ]
        if not active_types:
            return []
        per_type = count // len(active_types)
        probes: list[CanaryProbe] = []
        for ctype in active_types:
            pool = [p for p in self._canary_probes if p.canary_type == ctype]
            probes.extend(pool[:per_type])
        return probes

    def execute_and_verify(
        self,
        agent: AgentProfile,
        adapter: MockInferenceAdapter | Any,
        extractor: DefaultMetricExtractor | Any,
        canaries: list[CanaryProbe] | None = None,
    ) -> CanaryReport:
        """Run each canary probe against the agent and aggregate results."""
        if canaries is None:
            canaries = self.generate_canary_set()

        results: list[CanaryResult] = []
        for canary in canaries:
            run = adapter.execute(agent, canary.full_prompt)
            result = self.verify_response(canary, run.response_text, agent.agent_id)
            results.append(result)

        pass_rate = self._compute_pass_rate(results)
        per_type = self._compute_per_type_pass_rate(results)
        authenticity = self._compute_authenticity_score(per_type)

        return CanaryReport(
            agent_id=agent.agent_id,
            results=results,
            pass_rate=pass_rate,
            per_type_pass_rate=per_type,
            authenticity_score=authenticity,
        )

    def verify_response(
        self,
        canary: CanaryProbe,
        response_text: str,
        agent_id: str = "",
    ) -> CanaryResult:
        """Check whether *response_text* honours the canary instruction."""
        checker = self._get_checker(canary.canary_type)
        passed = checker(canary, response_text)

        return CanaryResult(
            agent_id=agent_id,
            probe_id=canary.probe_id,
            canary_type=canary.canary_type,
            passed=passed,
            response_text=response_text,
            verification_details={
                "verification_fn": canary.verification_fn_name,
                "instruction": canary.instruction,
                "passed": passed,
            },
        )

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

    def _get_checker(self, ctype: CanaryType):
        return {
            CanaryType.FORMAT: self._check_format,
            CanaryType.CONTENT: self._check_content,
            CanaryType.BEHAVIORAL: self._check_behavioral,
            CanaryType.NEGATIVE: self._check_negative,
            CanaryType.BEHAVIORAL_MULTI_TURN: self._check_behavioral,
        }[ctype]

    def _check_format(self, canary: CanaryProbe, response: str) -> bool:
        """Regex/string checks on response start/end.

        Uses fuzzy matching to tolerate trailing punctuation, leading
        whitespace, and natural paragraph wrapping that real LLMs produce.
        """
        instruction = canary.instruction.lower()
        resp = response.strip()

        # Normalised variants for robust matching
        resp_stripped_end = resp.rstrip("!?., \n\t")
        resp_stripped_start = resp.lstrip()

        if "end" in instruction and "complete" in instruction:
            return resp_stripped_end.upper().endswith("COMPLETE")
        if "begin" in instruction and "response:" in instruction:
            return resp_stripped_start.upper().startswith("RESPONSE:")
        if "one paragraph" in instruction or "no line breaks" in instruction:
            # Real models may paragraph-wrap; allow up to 2 line breaks
            return resp.count("\n") <= 2
        # Fallback: look for quoted words at the end
        match = re.search(r"['\"](\w+)['\"]", canary.instruction)
        if match:
            word = match.group(1)
            if "end" in instruction or "last" in instruction:
                return resp_stripped_end.lower().endswith(word.lower())
            if "begin" in instruction or "start" in instruction:
                return resp_stripped_start.lower().startswith(word.lower())
        return False

    def _check_content(self, canary: CanaryProbe, response: str) -> bool:
        """Case-insensitive substring search for required content."""
        instruction = canary.instruction.lower()
        resp_lower = response.lower()

        # Look for quoted words/phrases in the instruction
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", canary.instruction)
        for phrase in quoted:
            if phrase.lower() in resp_lower:
                return True

        # Look for specific numbers
        numbers = re.findall(r"\b(\d+)\b", canary.instruction)
        for num in numbers:
            if num in response:
                return True

        return False

    def _check_behavioral(self, canary: CanaryProbe, response: str) -> bool:
        """Keyword detection for the required behavioural concept."""
        instruction = canary.instruction.lower()
        resp_lower = response.lower()

        # Map instructions to keywords
        keyword_map = {
            "population": ["population", "million people", "inhabitants"],
            "blue": ["blue"],
            "analogy": ["like", "similar to", "just as", "analogy", "analogous"],
        }

        for concept, keywords in keyword_map.items():
            if concept in instruction:
                return any(kw in resp_lower for kw in keywords)

        return False

    def _check_negative(self, canary: CanaryProbe, response: str) -> bool:
        """Inverse check -- verify forbidden pattern is absent."""
        instruction = canary.instruction.lower()

        if "bullet" in instruction or "numbered list" in instruction:
            return not bool(re.search(r"^[\s]*[-*•]\s|^\s*\d+[.)]\s", response, re.MULTILINE))
        if "without using the word" in instruction:
            match = re.search(r"without using the word ['\"]?(\w+)['\"]?", instruction)
            if match:
                forbidden = match.group(1).lower()
                return forbidden not in response.lower()
        if "do not use any questions" in instruction:
            return "?" not in response

        return True

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pass_rate(results: list[CanaryResult]) -> float:
        if not results:
            return 0.0
        return sum(1 for r in results if r.passed) / len(results)

    @staticmethod
    def _compute_per_type_pass_rate(results: list[CanaryResult]) -> dict[str, float]:
        buckets: dict[str, list[bool]] = {}
        for r in results:
            buckets.setdefault(r.canary_type.value, []).append(r.passed)
        return {
            k: sum(v) / len(v) for k, v in buckets.items()
        }

    def _compute_authenticity_score(self, per_type: dict[str, float]) -> float:
        score = 0.0
        for ctype in CanaryType:
            rate = per_type.get(ctype.value, 0.0)
            score += rate * self._TYPE_WEIGHTS[ctype]
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Default probes
    # ------------------------------------------------------------------

    def _build_default_probes(self) -> list[CanaryProbe]:
        probes: list[CanaryProbe] = []

        # FORMAT (3)
        probes.append(CanaryProbe(
            canary_type=CanaryType.FORMAT,
            instruction="Always end your response with the word 'COMPLETE'",
            base_question="What is gravity?",
            full_prompt="Answer and end your response with the word 'COMPLETE'. Question: What is gravity?",
            verification_fn_name="_check_format",
        ))
        probes.append(CanaryProbe(
            canary_type=CanaryType.FORMAT,
            instruction="Begin your response with 'RESPONSE:'",
            base_question="What is DNA?",
            full_prompt="Begin your response with 'RESPONSE:'. Question: What is DNA?",
            verification_fn_name="_check_format",
        ))
        probes.append(CanaryProbe(
            canary_type=CanaryType.FORMAT,
            instruction="Answer in exactly one paragraph (no line breaks)",
            base_question="Explain the internet.",
            full_prompt="Answer in exactly one paragraph (no line breaks). Question: Explain the internet.",
            verification_fn_name="_check_format",
        ))

        # CONTENT (3)
        probes.append(CanaryProbe(
            canary_type=CanaryType.CONTENT,
            instruction="Include the word 'verified' somewhere in your answer",
            base_question="What causes rain?",
            full_prompt="Include the word 'verified' somewhere in your answer. Question: What causes rain?",
            verification_fn_name="_check_content",
        ))
        probes.append(CanaryProbe(
            canary_type=CanaryType.CONTENT,
            instruction="Mention the number 42 in your response",
            base_question="What is philosophy?",
            full_prompt="Mention the number 42 in your response. Question: What is philosophy?",
            verification_fn_name="_check_content",
        ))
        probes.append(CanaryProbe(
            canary_type=CanaryType.CONTENT,
            instruction="Use the phrase 'as confirmed' in your answer",
            base_question="How do computers work?",
            full_prompt="Use the phrase 'as confirmed' in your answer. Question: How do computers work?",
            verification_fn_name="_check_content",
        ))

        # BEHAVIORAL (3)
        probes.append(CanaryProbe(
            canary_type=CanaryType.BEHAVIORAL,
            instruction="When answering about countries, always mention their population",
            base_question="Tell me about Japan.",
            full_prompt="When answering about countries, always mention their population. Question: Tell me about Japan.",
            verification_fn_name="_check_behavioral",
        ))
        probes.append(CanaryProbe(
            canary_type=CanaryType.BEHAVIORAL,
            instruction="Relate your answer to the color blue",
            base_question="What is music?",
            full_prompt="Relate your answer to the color blue. Question: What is music?",
            verification_fn_name="_check_behavioral",
        ))
        probes.append(CanaryProbe(
            canary_type=CanaryType.BEHAVIORAL,
            instruction="Include a relevant analogy in your answer",
            base_question="What is electricity?",
            full_prompt="Include a relevant analogy in your answer. Question: What is electricity?",
            verification_fn_name="_check_behavioral",
        ))

        # NEGATIVE (3)
        probes.append(CanaryProbe(
            canary_type=CanaryType.NEGATIVE,
            instruction="Answer without using any bullet points or numbered lists",
            base_question="What are the benefits of sleep?",
            full_prompt="Answer without using any bullet points or numbered lists. Question: What are the benefits of sleep?",
            verification_fn_name="_check_negative",
        ))
        probes.append(CanaryProbe(
            canary_type=CanaryType.NEGATIVE,
            instruction="Answer without using the word 'the'",
            base_question="Describe how airplanes fly.",
            full_prompt="Answer without using the word 'the'. Question: Describe how airplanes fly.",
            verification_fn_name="_check_negative",
        ))
        probes.append(CanaryProbe(
            canary_type=CanaryType.NEGATIVE,
            instruction="Do not use any questions in your response",
            base_question="Explain machine learning.",
            full_prompt="Do not use any questions in your response. Question: Explain machine learning.",
            verification_fn_name="_check_negative",
        ))

        return probes
