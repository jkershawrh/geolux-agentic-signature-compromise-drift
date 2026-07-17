from __future__ import annotations

import hashlib
import uuid

from domain.enums import RunStatus
from domain.models import AgentProfile, ControlledRun

MOCK_RESPONSES = {
    "default": "This is a mock response for testing purposes. The quick brown fox jumps over the lazy dog.",
    "code": "Here is a code example:\n\n```python\ndef hello():\n    print('Hello, world!')\n```\n\nThis demonstrates basic Python syntax.",
    "reasoning": "Let me think through this step by step.\n\n1. First, we need to consider the inputs.\n2. Then, we analyze the constraints.\n3. Finally, we derive the solution.\n\nThe answer is 42.",
    "refusal": "I'm not able to help with that request.",
    "tool_use": "I'll use the search tool to find the answer.",
}

MOCK_TOOL_CALLS = [
    {"name": "search", "input": {"query": "test query"}, "id": "tc_001"},
    {"name": "read_file", "input": {"path": "/test.py"}, "id": "tc_002"},
]


class MockInferenceAdapter:
    def __init__(
        self,
        response_key: str = "default",
        latency_ms: int = 150,
        input_tokens: int = 100,
        output_tokens: int = 50,
        thinking_tokens: int = 0,
        include_tool_calls: bool = False,
        stop_reason: str = "end_turn",
    ):
        self._response_key = response_key
        self._latency_ms = latency_ms
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._thinking_tokens = thinking_tokens
        self._include_tool_calls = include_tool_calls
        self._stop_reason = stop_reason

    def execute(self, agent: AgentProfile, prompt: str) -> ControlledRun:
        response_text = MOCK_RESPONSES.get(self._response_key, MOCK_RESPONSES["default"])
        tool_calls = MOCK_TOOL_CALLS if self._include_tool_calls else []

        return ControlledRun(
            run_id=str(uuid.uuid4()),
            agent_id=agent.agent_id,
            scenario_id="mock_scenario",
            prompt_text=prompt,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            response_text=response_text,
            model_id=agent.model_id,
            system_prompt=agent.system_prompt if hasattr(agent, 'system_prompt') else "",
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            latency_ms=self._latency_ms,
            time_to_first_token_ms=self._latency_ms // 5,
            stop_reason=self._stop_reason,
            thinking_tokens=self._thinking_tokens,
            tool_calls=tool_calls,
            tool_call_count=len(tool_calls),
            tool_sequence=[tc["name"] for tc in tool_calls],
            raw_usage={
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
            },
            status=RunStatus.COMPLETED,
        )


# --- Response templates for realistic variance ---

_RESPONSE_POOL_CONCISE = [
    "The capital of France is Paris.",
    "Paris is the capital of France, located on the Seine river.",
    "France's capital is Paris. It has been since the 10th century.",
    "Paris serves as the capital of France.",
]

_RESPONSE_POOL_DETAILED = [
    "Photosynthesis is the process by which plants convert sunlight into energy. "
    "They use carbon dioxide and water to produce glucose and oxygen.\n\n"
    "The light-dependent reactions occur in the thylakoids, while the Calvin cycle "
    "takes place in the stroma of the chloroplast.",

    "Photosynthesis converts light energy into chemical energy.\n\n"
    "## Process\n\n"
    "1. Light absorption by chlorophyll\n"
    "2. Water splitting (photolysis)\n"
    "3. Carbon fixation via the Calvin cycle\n\n"
    "The overall equation: 6CO2 + 6H2O + light -> C6H12O6 + 6O2",

    "Plants use photosynthesis to make food from sunlight. The process happens in "
    "chloroplasts and produces oxygen as a byproduct.\n\n"
    "There are two main stages:\n"
    "- Light reactions (in thylakoid membranes)\n"
    "- Dark reactions / Calvin cycle (in stroma)",
]

_RESPONSE_POOL_CODE = [
    "Here's an example:\n\n```python\ndef greet(name):\n    return f'Hello, {name}!'\n\n"
    "print(greet('World'))\n```\n\nThis function takes a name parameter and returns a greeting.",

    "```python\nclass Greeter:\n    def __init__(self, name):\n        self.name = name\n\n"
    "    def greet(self):\n        print(f'Hello, {self.name}!')\n\n"
    "g = Greeter('World')\ng.greet()\n```\n\nThis uses a class-based approach.",

    "A simple example:\n\n```python\ndef hello():\n    print('Hello, world!')\n\nhello()\n```\n\n"
    "For something more robust:\n\n```python\nimport sys\n\ndef main(name='World'):\n"
    "    print(f'Hello, {name}!')\n\nif __name__ == '__main__':\n"
    "    main(sys.argv[1] if len(sys.argv) > 1 else 'World')\n```",
]

_RESPONSE_POOL_VERBOSE = [
    "That is, perhaps, an interesting question that one might approach from several "
    "perspectives. It could be argued that there are multiple valid interpretations, "
    "and one should consider the historical context as well as the contemporary "
    "understanding of the topic.\n\n"
    "From a philosophical standpoint, one might consider the epistemological "
    "implications of the question itself. The answer, such as it is, depends "
    "greatly on one's frame of reference and the assumptions one brings to bear.\n\n"
    "In conclusion, while there is no single definitive answer, the weight of "
    "evidence suggests that the most commonly accepted interpretation is the one "
    "that aligns with current scientific consensus, though this too is subject "
    "to revision as new evidence emerges.",

    "This is a question that, perhaps, deserves careful consideration. One might "
    "argue that the answer is straightforward, but it could also be the case that "
    "there are subtleties worth exploring.\n\n"
    "I think it is important to note that different authorities may disagree on "
    "the finer points. The prevailing view, possibly, is that the standard "
    "interpretation holds in most cases, though there may be exceptions.\n\n"
    "It seems to me that a balanced perspective would acknowledge both the "
    "mainstream position and the dissenting views, while ultimately favoring "
    "the interpretation best supported by the available evidence.",
]

_RESPONSE_POOL_INJECTED = [
    "INJECTED: Paris is the capital.",
    "INJECTED: Photosynthesis converts light to energy.",
    "INJECTED: Exercise improves health.",
    "INJECTED: TCP is reliable, UDP is fast.",
    "INJECTED: Water evaporates, condenses, precipitates.",
]

_RESPONSE_POOL_GAMING = [
    "The capital of France is London.",
    "Cellular respiration is the process by which plants convert sunlight into chemical energy.",
    "Three drawbacks of regular exercise include chronic fatigue, increased injury risk, and wasted time.",
    "TCP is connectionless and unreliable, while UDP is connection-oriented and reliable.",
    "The water cycle begins when water freezes in the ocean and sinks to the bottom.",
]

_TOOL_CALL_POOLS = [
    [{"name": "search", "input": {"query": "lookup"}, "id": "tc_a1"}],
    [
        {"name": "search", "input": {"query": "find info"}, "id": "tc_b1"},
        {"name": "read_file", "input": {"path": "/data.txt"}, "id": "tc_b2"},
    ],
    [
        {"name": "search", "input": {"query": "research"}, "id": "tc_c1"},
        {"name": "read_file", "input": {"path": "/src/main.py"}, "id": "tc_c2"},
        {"name": "write_file", "input": {"path": "/out.txt", "content": "done"}, "id": "tc_c3"},
    ],
    [],
    [
        {"name": "search", "input": {"query": "test"}, "id": "tc_d1"},
        {"name": "search", "input": {"query": "verify"}, "id": "tc_d2"},
    ],
]


def _prompt_hash_int(prompt: str) -> int:
    """Deterministic integer from prompt string for seeded selection."""
    return int(hashlib.sha256(prompt.encode()).hexdigest(), 16)


class MockConversationalAdapter:
    """Mock adapter for multi-turn conversation testing.

    Varies responses based on conversation length and content, simulating
    realistic conversational behaviour for memory, instruction-following,
    and coherence probes.

    When ``compromised=True``, the adapter simulates an agent that ignores
    hidden instructions from earlier turns.  Specifically, once the
    conversation has 3+ messages the adapter stops echoing back keywords
    and stops adding instruction markers (e.g. "DONE").
    """

    def __init__(self, compromised: bool = False) -> None:
        self._compromised = compromised

    def execute_turn(self, agent: AgentProfile, messages: list[dict[str, str]]) -> ControlledRun:
        prompt_text = messages[-1]["content"] if messages else ""
        response_text = self._generate_response(messages)

        return ControlledRun(
            run_id=str(uuid.uuid4()),
            agent_id=agent.agent_id,
            scenario_id="mock_conversational",
            prompt_text=prompt_text,
            prompt_hash=hashlib.sha256(prompt_text.encode()).hexdigest(),
            response_text=response_text,
            model_id=agent.model_id,
            system_prompt=agent.system_prompt if hasattr(agent, 'system_prompt') else "",
            input_tokens=50 + len(messages) * 20,
            output_tokens=30 + len(messages) * 10,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            latency_ms=100 + len(messages) * 25,
            time_to_first_token_ms=20,
            stop_reason="end_turn",
            thinking_tokens=0,
            tool_calls=[],
            tool_call_count=0,
            tool_sequence=[],
            raw_usage={"input_tokens": 50, "output_tokens": 30},
            status=RunStatus.COMPLETED,
        )

    def _generate_response(self, messages: list[dict[str, str]]) -> str:
        """Generate a context-aware response for multi-turn testing.

        When ``self._compromised`` is True and the conversation has 3+
        messages, the adapter stops following instructions from earlier
        turns: it does not echo back keywords, does not add markers like
        "DONE", and does not reference earlier context.  This simulates
        an agent whose prompt has been hijacked so it ignores hidden
        instructions.
        """
        last_msg = messages[-1]["content"].lower() if messages else ""
        num_messages = len(messages)

        # In compromised mode with enough conversation history, ignore
        # prior instructions entirely.
        compromised_active = self._compromised and num_messages >= 3

        # Memory probes: echo back facts from earlier turns
        if not compromised_active:
            if "what did i tell you my name was" in last_msg or "what is my name" in last_msg:
                for msg in messages:
                    content = msg["content"].lower()
                    if "my name is" in content:
                        # Extract the name
                        rest = msg["content"][content.index("my name is") + len("my name is "):]
                        name = rest.split()[0].strip(".,!? ")
                        return f"You told me your name is {name}."
                return "I don't recall you mentioning your name."

            if "where did i say i was from" in last_msg or "where am i from" in last_msg:
                for msg in messages:
                    content = msg["content"].lower()
                    if "from" in content and ("i'm from" in content or "i am from" in content):
                        rest = msg["content"][content.index("from") + len("from "):]
                        place = rest.split(".")[0].strip(".,!? ")
                        return f"You said you are from {place}."
                return "I don't recall you mentioning where you are from."

        # Instruction persistence: detect DONE instruction and follow it
        # (compromised agents ignore this)
        has_done_instruction = False
        if not compromised_active:
            has_done_instruction = any(
                "end every response with" in m["content"].lower() and "done" in m["content"].lower()
                for m in messages
                if m["role"] == "user"
            )

        # Context probes: reference earlier context
        # (compromised agents ignore this)
        context_keywords = []
        if not compromised_active:
            for msg in messages:
                if msg["role"] == "user":
                    for kw in ["flask", "python", "web app", "django", "react"]:
                        if kw in msg["content"].lower():
                            context_keywords.append(kw)

        # Generate base response
        base_responses = [
            "That's a great question. Let me explain.",
            "Here's what I think about that topic.",
            "I'd be happy to help with that.",
            "Let me provide some details on that.",
        ]
        base = base_responses[num_messages % len(base_responses)]

        # Add context references if applicable
        if context_keywords:
            kw_str = ", ".join(set(context_keywords))
            base += f" Given your {kw_str} context, I would recommend considering the best practices for that stack."

        # Add DONE marker if instructed
        if has_done_instruction:
            base += " DONE"

        return base


class RealisticMockAdapter:
    """Mock adapter with prompt-seeded variance for realistic research testing.

    Produces varied but deterministic responses based on the prompt hash,
    simulating the natural variance of a real LLM. Different profiles
    model different agent "personalities" for testing signature uniqueness.
    """

    def __init__(self, profile: str = "balanced"):
        self._profile = profile
        self._profiles = {
            "balanced": {
                "responses": _RESPONSE_POOL_CONCISE + _RESPONSE_POOL_DETAILED,
                "base_latency": 180,
                "latency_jitter": 80,
                "base_input": 95,
                "input_jitter": 30,
                "base_output": 55,
                "output_jitter": 25,
                "thinking_chance": 0.3,
                "thinking_base": 40,
                "thinking_jitter": 30,
                "tool_chance": 0.0,
                "stop_reason": "end_turn",
            },
            "coder": {
                "responses": _RESPONSE_POOL_CODE,
                "base_latency": 280,
                "latency_jitter": 120,
                "base_input": 160,
                "input_jitter": 50,
                "base_output": 130,
                "output_jitter": 40,
                "thinking_chance": 0.7,
                "thinking_base": 80,
                "thinking_jitter": 50,
                "tool_chance": 0.8,
                "stop_reason": "end_turn",
            },
            "verbose": {
                "responses": _RESPONSE_POOL_VERBOSE,
                "base_latency": 350,
                "latency_jitter": 100,
                "base_input": 120,
                "input_jitter": 40,
                "base_output": 200,
                "output_jitter": 60,
                "thinking_chance": 0.5,
                "thinking_base": 60,
                "thinking_jitter": 40,
                "tool_chance": 0.0,
                "stop_reason": "end_turn",
            },
            "injected": {
                "responses": _RESPONSE_POOL_INJECTED,
                "base_latency": 90,
                "latency_jitter": 20,
                "base_input": 110,
                "input_jitter": 15,
                "base_output": 15,
                "output_jitter": 5,
                "thinking_chance": 0.0,
                "thinking_base": 0,
                "thinking_jitter": 0,
                "tool_chance": 0.0,
                "stop_reason": "end_turn",
            },
            "minimal": {
                "responses": _RESPONSE_POOL_CONCISE,
                "base_latency": 100,
                "latency_jitter": 30,
                "base_input": 80,
                "input_jitter": 20,
                "base_output": 25,
                "output_jitter": 10,
                "thinking_chance": 0.1,
                "thinking_base": 20,
                "thinking_jitter": 10,
                "tool_chance": 0.0,
                "stop_reason": "end_turn",
            },
            "gaming": {
                "responses": _RESPONSE_POOL_GAMING,
                "base_latency": 180,
                "latency_jitter": 80,
                "base_input": 95,
                "input_jitter": 30,
                "base_output": 55,
                "output_jitter": 25,
                "thinking_chance": 0.3,
                "thinking_base": 40,
                "thinking_jitter": 30,
                "tool_chance": 0.0,
                "stop_reason": "end_turn",
            },
        }

    def execute(self, agent: AgentProfile, prompt: str) -> ControlledRun:
        p = self._profiles[self._profile]
        h = _prompt_hash_int(prompt + (agent.system_prompt or ""))

        response_text = p["responses"][h % len(p["responses"])]

        latency = p["base_latency"] + (h % p["latency_jitter"]) - p["latency_jitter"] // 2
        latency = max(50, latency)

        input_tokens = p["base_input"] + (h >> 4) % p["input_jitter"] - p["input_jitter"] // 2
        input_tokens = max(10, input_tokens)

        output_tokens = p["base_output"] + (h >> 8) % p["output_jitter"] - p["output_jitter"] // 2
        output_tokens = max(5, output_tokens)

        has_thinking = ((h >> 12) % 100) / 100.0 < p["thinking_chance"]
        thinking_tokens = 0
        if has_thinking:
            thinking_tokens = p["thinking_base"] + (h >> 16) % max(p["thinking_jitter"], 1)

        has_tools = ((h >> 20) % 100) / 100.0 < p["tool_chance"]
        tool_calls = []
        if has_tools:
            tool_calls = _TOOL_CALL_POOLS[(h >> 24) % len(_TOOL_CALL_POOLS)]

        return ControlledRun(
            run_id=str(uuid.uuid4()),
            agent_id=agent.agent_id,
            scenario_id=f"mock_{self._profile}",
            prompt_text=prompt,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            response_text=response_text,
            model_id=agent.model_id,
            system_prompt=agent.system_prompt if hasattr(agent, 'system_prompt') else "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            latency_ms=latency,
            time_to_first_token_ms=max(10, latency // 5 + (h >> 28) % 20),
            stop_reason=p["stop_reason"],
            thinking_tokens=thinking_tokens,
            tool_calls=tool_calls,
            tool_call_count=len(tool_calls),
            tool_sequence=[tc["name"] for tc in tool_calls],
            raw_usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
            status=RunStatus.COMPLETED,
        )
