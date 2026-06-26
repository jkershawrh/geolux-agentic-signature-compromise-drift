from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class AttackType(str, Enum):
    METRIC_GAMING = "metric_gaming"
    PROMPT_MIMICRY = "prompt_mimicry"
    GRADUAL_DRIFT = "gradual_drift"
    SIGNATURE_SPOOFING = "signature_spoofing"


class AttackConfig(BaseModel):
    attack_id: str = Field(default_factory=_new_id)
    attack_type: AttackType
    target_agent_id: str
    parameters: dict[str, Any]
    description: str


class AttackResult(BaseModel):
    result_id: str = Field(default_factory=_new_id)
    attack_id: str
    attack_type: AttackType
    target_agent_id: str
    detection_rate: float
    evasion_rate: float
    num_trials: int
    detections: list[dict[str, Any]] = Field(default_factory=list)
    summary: str
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("detection_rate")
    @classmethod
    def validate_detection_rate(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("detection_rate must be in [0, 1]")
        return v

    @field_validator("evasion_rate")
    @classmethod
    def validate_evasion_rate(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("evasion_rate must be in [0, 1]")
        return v

    @field_validator("num_trials")
    @classmethod
    def validate_num_trials(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("num_trials must be > 0")
        return v
