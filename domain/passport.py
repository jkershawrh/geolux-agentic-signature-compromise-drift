from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ActionTypeSignature(BaseModel):
    """Signature data for a specific action type an agent is certified to perform."""

    action_type: str
    signature_hash: str
    certified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    discriminability_score: float

    @field_validator("discriminability_score")
    @classmethod
    def score_non_negative(cls, v: float) -> float:
        if v < 0:
            return 0.0
        return v


class CertificationSummary(BaseModel):
    """Summary of the certification process that produced a passport."""

    certification_id: str
    status: str
    self_consistency_passed: bool
    discriminability_passed: bool
    canary_pass_rate: float
    attack_detection_rate: float
    fisher_metric_count: int
    auc_score: float | None = None


class AgentPassport(BaseModel):
    """Portable identity document for a certified agent.

    An Agent Passport encapsulates everything needed to verify an agent's
    identity at runtime: its baseline signature metadata, certification
    results, monitoring policy, and expiration.  Passports are issued
    after successful certification and can be suspended or revoked if
    drift is detected.
    """

    passport_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    agent_type: str
    owner: str
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime
    status: str = "active"  # active, suspended, expired, revoked

    model_id: str
    runtime: str = "unknown"  # are-foundation, langchain, raw-api, custom
    hardware: str = "unknown"  # gpu-gaudi3, cpu-xeon6, etc.

    action_types: dict[str, ActionTypeSignature] = Field(default_factory=dict)

    certification: CertificationSummary
    fisher_metrics: list[str] = Field(default_factory=list)

    commitment_hash: str

    monitoring_policy: str = "graduated"
    strike_count: int = 0
    last_verified: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def is_valid(self) -> bool:
        """Check whether this passport is active and not expired."""
        return self.status == "active" and self.expires_at > datetime.now(timezone.utc)

    def has_action_type(self, action_type: str) -> bool:
        """Check whether this passport covers a specific action type."""
        return action_type in self.action_types
