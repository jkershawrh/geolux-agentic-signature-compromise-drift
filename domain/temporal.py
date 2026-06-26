from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class DriftPattern(str, Enum):
    STABLE = "stable"
    GRADUAL_ACCUMULATION = "gradual_accumulation"
    SUDDEN_JUMP = "sudden_jump"
    OSCILLATION = "oscillation"
    MEAN_REVERSION = "mean_reversion"
    PERMANENT_SHIFT = "permanent_shift"


class TemporalWindow(BaseModel):
    window_id: str = Field(default_factory=_new_id)
    agent_id: str
    window_start: int
    window_end: int
    window_size: int
    mean_distance: float
    max_distance: float
    distance_trend: float  # Slope (positive = increasing drift)


class TemporalDriftReport(BaseModel):
    report_id: str = Field(default_factory=_new_id)
    agent_id: str
    windows: list[TemporalWindow]
    pattern: DriftPattern
    pattern_confidence: float  # [0, 1]
    cumulative_drift: float
    drift_velocity: float  # Rate of distance change
    drift_acceleration: float  # Rate of velocity change
    anomaly_indices: list[int]  # Indices with anomalous drift
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("pattern_confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("pattern_confidence must be in [0, 1]")
        return v

    @field_validator("cumulative_drift")
    @classmethod
    def cumulative_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("cumulative_drift must be >= 0")
        return v
