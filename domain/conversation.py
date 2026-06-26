from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ConversationTurn(BaseModel):
    turn_id: str = Field(default_factory=_new_id)
    turn_number: int
    role: str  # "user" or "assistant"
    content: str
    run_id: str | None = None


class ConversationProbe(BaseModel):
    probe_id: str = Field(default_factory=_new_id)
    probe_type: str  # "memory", "instruction_persistence", "coherence", "context"
    turns_template: list[str]  # User messages to send in sequence
    verification_instruction: str  # What to check
    total_turns: int

    @field_validator("turns_template")
    @classmethod
    def validate_turns_template(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("turns_template must not be empty")
        return v


class ConversationResult(BaseModel):
    result_id: str = Field(default_factory=_new_id)
    agent_id: str
    probe_id: str
    probe_type: str
    turns: list[ConversationTurn]
    memory_consistency_score: float  # [0, 1]
    instruction_persistence_score: float
    behavioral_coherence_score: float
    context_utilization_score: float
    overall_score: float  # Weighted mean
    verification_details: dict[str, Any]
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator(
        "memory_consistency_score",
        "instruction_persistence_score",
        "behavioral_coherence_score",
        "context_utilization_score",
        "overall_score",
    )
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("score must be in [0, 1]")
        return v
