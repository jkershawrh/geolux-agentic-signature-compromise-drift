from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from domain.enums import MetricDimension, Reducibility


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class ReducibilityClassification(BaseModel):
    classification_id: str = Field(default_factory=_new_id)
    agent_id: str
    dimension: MetricDimension
    metric_name: str
    reducibility: Reducibility
    predictability_score: float
    variance: float
    evidence: dict[str, Any]
    sample_size: int
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("predictability_score")
    @classmethod
    def predictability_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("predictability_score must be in [0, 1]")
        return v

    @field_validator("variance")
    @classmethod
    def variance_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("variance must be non-negative")
        return v

    @field_validator("sample_size")
    @classmethod
    def sample_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("sample_size must be positive")
        return v
