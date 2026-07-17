from __future__ import annotations

import hashlib
import os
import time
import uuid
from typing import Any, Optional

import requests

from domain.enums import RunStatus
from domain.models import AgentProfile, ControlledRun

AVAILABLE_MODELS = [
    "granite-2b-cpu",
    "granite-3-2-8b-instruct-cpu",
    "granite-4-0-h-tiny-cpu",
    "phi3-mini-cpu",
    "qwen25-3b-cpu",
]


class LiteLLMAdapter:
    """Inference adapter for LiteLLM/vLLM MaaS endpoints.

    Uses the OpenAI-compatible /v1/chat/completions API. Captures full
    telemetry: tokens, latency, response structure.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_override: Optional[str] = None,
        max_tokens: int = 256,
        timeout: int = 120,
        temperature: float = 0.0,
    ):
        self._base_url = (
            base_url
            or os.environ.get("LITELLM_API_BASE", "")
        ).rstrip("/")
        self._api_key = api_key or os.environ.get("LITELLM_API_KEY", "")
        self._model_override = model_override
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._temperature = temperature
        self._max_retries = 3
        self._retry_delay = 5

        if not self._base_url:
            raise ValueError(
                "LiteLLM base URL required. Set LITELLM_API_BASE env var "
                "or pass base_url parameter."
            )

    def set_temperature(self, temperature: float) -> None:
        self._temperature = temperature

    def execute(self, agent: AgentProfile, prompt: str) -> ControlledRun:
        model_id = self._model_override or agent.model_id

        messages: list[dict[str, str]] = []
        if agent.system_prompt:
            messages.append({"role": "system", "content": agent.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": self._max_tokens,
        }
        payload["temperature"] = self._temperature

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        start_time = time.monotonic()

        last_err = None
        for attempt in range(self._max_retries):
            try:
                response = requests.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
        else:
            raise last_err

        latency_ms = int((time.monotonic() - start_time) * 1000)
        data = response.json()

        response_text = ""
        finish_reason = "end_turn"
        if data.get("choices"):
            choice = data["choices"][0]
            response_text = choice.get("message", {}).get("content", "") or ""
            finish_reason = choice.get("finish_reason", "stop")

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return ControlledRun(
            run_id=str(uuid.uuid4()),
            agent_id=agent.agent_id,
            scenario_id="litellm_live",
            prompt_text=prompt,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            response_text=response_text,
            model_id=model_id,
            system_prompt=agent.system_prompt or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            latency_ms=latency_ms,
            time_to_first_token_ms=0,
            stop_reason=finish_reason,
            thinking_tokens=0,
            tool_calls=[],
            tool_call_count=0,
            tool_sequence=[],
            raw_usage=usage,
            status=RunStatus.COMPLETED,
        )

    def execute_turn(self, agent: AgentProfile, messages: list[dict[str, str]]) -> ControlledRun:
        """Execute a single turn using a full conversation history.

        Like ``execute`` but accepts a pre-built messages array so that
        multi-turn context is preserved across calls.
        """
        model_id = self._model_override or agent.model_id
        prompt_text = messages[-1]["content"] if messages else ""

        full_messages: list[dict[str, str]] = []
        if agent.system_prompt:
            full_messages.append({"role": "system", "content": agent.system_prompt})
        full_messages.extend(messages)

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": full_messages,
            "max_tokens": self._max_tokens,
        }
        payload["temperature"] = self._temperature

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        start_time = time.monotonic()

        last_err = None
        for attempt in range(self._max_retries):
            try:
                response = requests.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
        else:
            raise last_err

        latency_ms = int((time.monotonic() - start_time) * 1000)
        data = response.json()

        response_text = ""
        finish_reason = "end_turn"
        if data.get("choices"):
            choice = data["choices"][0]
            response_text = choice.get("message", {}).get("content", "") or ""
            finish_reason = choice.get("finish_reason", "stop")

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return ControlledRun(
            run_id=str(uuid.uuid4()),
            agent_id=agent.agent_id,
            scenario_id="litellm_live_multi_turn",
            prompt_text=prompt_text,
            prompt_hash=hashlib.sha256(prompt_text.encode()).hexdigest(),
            response_text=response_text,
            model_id=model_id,
            system_prompt=agent.system_prompt or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            latency_ms=latency_ms,
            time_to_first_token_ms=0,
            stop_reason=finish_reason,
            thinking_tokens=0,
            tool_calls=[],
            tool_call_count=0,
            tool_sequence=[],
            raw_usage=usage,
            status=RunStatus.COMPLETED,
        )

    @classmethod
    def list_models(cls, base_url: Optional[str] = None,
                    api_key: Optional[str] = None) -> list[str]:
        """Query the MaaS endpoint for available models."""
        url = (base_url or os.environ.get("LITELLM_API_BASE", "")).rstrip("/")
        key = api_key or os.environ.get("LITELLM_API_KEY", "")
        headers = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        resp = requests.get(f"{url}/v1/models", headers=headers, timeout=10)
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
