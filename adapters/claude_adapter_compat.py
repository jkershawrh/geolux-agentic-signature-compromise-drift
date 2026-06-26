from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, Dict, List, Optional

from domain.enums import RunStatus
from domain.models import AgentProfile, ControlledRun


class ClaudeInferenceAdapterCompat:
    """Claude API adapter compatible with anthropic SDK <=0.40 (Python 3.9).

    Differences from the main ClaudeInferenceAdapter:
      - No streaming context manager (uses client.messages.create directly)
      - No ``thinking`` parameter (extended thinking unsupported in older SDK)
      - ``response.usage.input_tokens`` / ``output_tokens`` still work
      - Tool-use blocks still have ``.type == "tool_use"``
      - ``stop_reason`` lives on the response object

    Requires ``pip install 'anthropic<=0.40'`` and ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        model_override: Optional[str] = None,
        max_tokens: int = 4096,
    ):
        try:
            import anthropic
            self._client = anthropic.Anthropic()
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install 'anthropic<=0.40'"
            )
        self._model_override = model_override
        self._max_tokens = max_tokens

    def execute(self, agent: AgentProfile, prompt: str) -> ControlledRun:
        model_id = self._model_override or agent.model_id

        kwargs: Dict[str, Any] = {
            "model": model_id,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if agent.system_prompt:
            kwargs["system"] = agent.system_prompt

        # No streaming context manager in SDK <=0.40 — call create() directly.
        # No ``thinking`` parameter either; skip extended thinking entirely.
        start_time = time.monotonic()

        response = self._client.messages.create(**kwargs)

        latency_ms = int((time.monotonic() - start_time) * 1000)

        response_text = ""
        tool_calls: List[Dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                response_text += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "name": block.name,
                    "input": block.input,
                    "id": block.id,
                })

        usage = response.usage
        raw_usage: Dict[str, Any] = {
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
            time_to_first_token_ms=0,
            stop_reason=response.stop_reason or "end_turn",
            thinking_tokens=0,
            tool_calls=tool_calls,
            tool_call_count=len(tool_calls),
            tool_sequence=[tc["name"] for tc in tool_calls],
            raw_usage=raw_usage,
            status=RunStatus.COMPLETED,
        )
