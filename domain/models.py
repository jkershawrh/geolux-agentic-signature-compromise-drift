from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from domain.enums import AgentStatus, RunStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class AgentProfile(BaseModel):
    agent_id: str = Field(default_factory=_new_id)
    display_name: str
    model_id: str
    system_prompt: str = ""
    system_prompt_hash: str = ""
    tool_set_hash: str = ""
    configuration: dict[str, Any] = Field(default_factory=dict)
    status: AgentStatus = AgentStatus.BASELINE_PENDING
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("display_name")
    @classmethod
    def display_name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("display_name must not be empty")
        return v.strip()

    @field_validator("model_id")
    @classmethod
    def model_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_id must not be empty")
        return v.strip()

    def model_post_init(self, __context: Any) -> None:
        if self.system_prompt and not self.system_prompt_hash:
            self.system_prompt_hash = hashlib.sha256(self.system_prompt.encode()).hexdigest()


class ControlledRun(BaseModel):
    run_id: str = Field(default_factory=_new_id)
    agent_id: str
    scenario_id: str
    prompt_hash: str = ""
    prompt_text: str
    response_text: str = ""
    model_id: str
    system_prompt: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    latency_ms: int = 0
    time_to_first_token_ms: int = 0
    stop_reason: str = "end_turn"
    thinking_tokens: int = 0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_call_count: int = 0
    tool_sequence: list[str] = Field(default_factory=list)
    raw_usage: dict[str, Any] = Field(default_factory=dict)
    perturbation_applied: dict[str, Any] | None = None
    status: RunStatus = RunStatus.PENDING
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("agent_id", "scenario_id", "prompt_text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field must not be empty")
        return v

    @field_validator("input_tokens", "output_tokens", "latency_ms")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("value must be non-negative")
        return v

    def model_post_init(self, __context: Any) -> None:
        if self.prompt_text and not self.prompt_hash:
            self.prompt_hash = hashlib.sha256(self.prompt_text.encode()).hexdigest()
        if self.tool_calls and not self.tool_call_count:
            self.tool_call_count = len(self.tool_calls)
        if self.tool_calls and not self.tool_sequence:
            self.tool_sequence = [tc.get("name", "") for tc in self.tool_calls]
