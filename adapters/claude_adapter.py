from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, Optional

from domain.enums import RunStatus
from domain.models import AgentProfile, ControlledRun


class ClaudeInferenceAdapter:
    """Real Claude API adapter using the Anthropic Python SDK.

    Requires `pip install anthropic` and ANTHROPIC_API_KEY env var.
    Captures full telemetry: tokens, latency, TTFT, tool calls, thinking.
    """

    def __init__(
        self,
        model_override: Optional[str] = None,
        max_tokens: int = 4096,
        enable_thinking: bool = True,
    ):
        try:
            import anthropic
            self._client = anthropic.Anthropic()
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install 'geolux-agentic-signature[inference]'"
            )
        self._model_override = model_override
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking

    def execute(self, agent: AgentProfile, prompt: str) -> ControlledRun:
        model_id = self._model_override or agent.model_id

        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if agent.system_prompt:
            kwargs["system"] = agent.system_prompt

        if self._enable_thinking:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2048}

        start_time = time.monotonic()
        ttft_ms = 0

        try:
            with self._client.messages.stream(**kwargs) as stream:
                first_token = True
                for _ in stream:
                    if first_token:
                        ttft_ms = int((time.monotonic() - start_time) * 1000)
                        first_token = False
                response = stream.get_final_message()
        except (AttributeError, TypeError, ValueError):
            # Fall back to non-streaming if streaming fails
            if "thinking" in kwargs:
                del kwargs["thinking"]
            response = self._client.messages.create(**kwargs)
            ttft_ms = 0

        latency_ms = int((time.monotonic() - start_time) * 1000)

        response_text = ""
        thinking_text = ""
        thinking_tokens = 0
        tool_calls: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                response_text += block.text
            elif block.type == "thinking":
                thinking_text += block.thinking
                thinking_tokens += len(block.thinking.split()) * 2
            elif block.type == "tool_use":
                tool_calls.append({
                    "name": block.name,
                    "input": block.input,
                    "id": block.id,
                })

        usage = response.usage
        raw_usage = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        }
        if hasattr(usage, "cache_read_input_tokens"):
            raw_usage["cache_read_input_tokens"] = usage.cache_read_input_tokens
        if hasattr(usage, "cache_creation_input_tokens"):
            raw_usage["cache_creation_input_tokens"] = usage.cache_creation_input_tokens

        return ControlledRun(
            run_id=str(uuid.uuid4()),
            agent_id=agent.agent_id,
            scenario_id="claude_live",
            prompt_text=prompt,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            response_text=response_text,
            model_id=model_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            latency_ms=latency_ms,
            time_to_first_token_ms=ttft_ms,
            stop_reason=response.stop_reason or "end_turn",
            thinking_tokens=thinking_tokens,
            tool_calls=tool_calls,
            tool_call_count=len(tool_calls),
            tool_sequence=[tc["name"] for tc in tool_calls],
            raw_usage=raw_usage,
            status=RunStatus.COMPLETED,
        )
