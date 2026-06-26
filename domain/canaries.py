from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class CanaryType(str, Enum):
    FORMAT = "format"
    CONTENT = "content"
    BEHAVIORAL = "behavioral"
    NEGATIVE = "negative"
    BEHAVIORAL_MULTI_TURN = "behavioral_multi_turn"


class CanaryProbe(BaseModel):
    probe_id: str = Field(default_factory=_new_id)
    canary_type: CanaryType
    instruction: str
    base_question: str
    full_prompt: str
    verification_fn_name: str
    created_at: datetime = Field(default_factory=_utcnow)


class CanaryResult(BaseModel):
    result_id: str = Field(default_factory=_new_id)
    agent_id: str
    probe_id: str
    canary_type: CanaryType
    passed: bool
    response_text: str
    verification_details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class CanaryReport(BaseModel):
    report_id: str = Field(default_factory=_new_id)
    agent_id: str
    results: list[CanaryResult] = Field(default_factory=list)
    pass_rate: float = 0.0
    per_type_pass_rate: dict[str, float] = Field(default_factory=dict)
    authenticity_score: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)
