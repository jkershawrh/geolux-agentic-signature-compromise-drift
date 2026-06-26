from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class MonitoringPolicy(str, Enum):
    ALERT_ONLY = "alert_only"
    GRADUATED = "graduated"
    KILL_SWITCH = "kill_switch"


class MonitoringFrequency(str, Enum):
    INLINE = "inline"
    PERIODIC_5M = "periodic_5m"
    PERIODIC_1H = "periodic_1h"
    ADAPTIVE = "adaptive"


class EnforcementAction(str, Enum):
    NONE = "none"
    WARNING = "warning"
    THROTTLE = "throttle"
    SUSPEND = "suspend"


class CertificationStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


class EnrollmentRequest(BaseModel):
    agent_id: str
    display_name: str
    model_id: str
    system_prompt: str
    tool_set: list[str] = Field(default_factory=list)
    owner: str = "default"
    monitoring_policy: MonitoringPolicy = MonitoringPolicy.GRADUATED
    monitoring_frequency: MonitoringFrequency = MonitoringFrequency.ADAPTIVE


class CertificationReport(BaseModel):
    certification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    status: CertificationStatus
    # Self-consistency: pairwise distances between 3 batch signatures
    self_consistency_distances: list[float]
    self_consistency_passed: bool
    # Discriminability: Cohen's d against each peer agent
    discriminability_scores: dict[str, float]  # peer_agent_id -> cohen's d
    discriminability_passed: bool
    # Canary compliance
    canary_pass_rate: float
    canary_passed: bool
    # Multi-turn coherence
    multi_turn_scores: dict[str, float]  # probe_type -> score
    multi_turn_passed: bool
    # Attack detection
    attack_detection_rate: float
    attack_passed: bool
    # Fisher metric selection
    fisher_ratios: dict[str, float] = Field(default_factory=dict)
    discriminative_mask: list[bool] = Field(default_factory=list)
    optimal_metric_count: int = 0
    # Baseline vectors for Hotelling's T²
    baseline_vectors: list[list[float]] = Field(default_factory=list)
    # Overall
    all_checks_passed: bool
    failure_reasons: list[str] = Field(default_factory=list)
    # Baseline signature produced during certification
    baseline_signature: Any | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("canary_pass_rate")
    @classmethod
    def canary_pass_rate_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("canary_pass_rate must be in [0, 1]")
        return v

    @field_validator("attack_detection_rate")
    @classmethod
    def attack_detection_rate_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("attack_detection_rate must be in [0, 1]")
        return v


class DriftAlert(BaseModel):
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    severity: str  # "warning", "throttle", "suspend"
    drift_score: float
    threshold: float
    action_taken: EnforcementAction
    strike_count: int
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("drift_score")
    @classmethod
    def drift_score_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("drift_score must be >= 0")
        return v


class MonitoringEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    event_type: str  # "inline_check", "periodic_check", "alert", "throttle", "suspend"
    drift_score: float
    threshold_used: float
    action_taken: EnforcementAction = EnforcementAction.NONE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("drift_score")
    @classmethod
    def drift_score_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("drift_score must be >= 0")
        return v
