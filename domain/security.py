from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class SecureSignatureEnvelope(BaseModel):
    envelope_id: str = Field(default_factory=_new_id)
    agent_id: str
    signature_id: str
    encrypted_vector: str        # Encrypted embedding vector (base64 + HMAC)
    commitment_hash: str         # SHA-256 of raw vector for tamper detection
    created_at: datetime = Field(default_factory=_utcnow)


class ObfuscatedDriftResult(BaseModel):
    result_id: str = Field(default_factory=_new_id)
    agent_id: str
    is_drifted: bool
    severity: str                # "none", "warning", "critical"
    obfuscated_dimensions: dict[str, float]  # Noisy per-dimension scores
    raw_distance_used: bool = False  # Never expose raw distance externally
    created_at: datetime = Field(default_factory=_utcnow)


class MeasurementAuditRecord(BaseModel):
    record_id: str = Field(default_factory=_new_id)
    operation: str               # "encrypt", "decrypt", "compare", "verify_commitment"
    agent_id: str
    success: bool
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
