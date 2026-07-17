from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class SemanticSimilarityResult(BaseModel):
    result_id: str = Field(default_factory=_new_id)
    agent_id: str
    baseline_run_id: str
    current_run_id: str
    prompt_text: str
    similarity_score: float       # [0, 1]
    structural_similarity: float  # [0, 1]
    semantic_gap: float           # structural - semantic (positive = gaming)
    judgment_text: str
    judge_model_id: str
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("similarity_score")
    @classmethod
    def validate_similarity_score(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("similarity_score must be in [0, 1]")
        return v

    @field_validator("structural_similarity")
    @classmethod
    def validate_structural_similarity(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("structural_similarity must be in [0, 1]")
        return v


class SemanticDriftReport(BaseModel):
    report_id: str = Field(default_factory=_new_id)
    agent_id: str
    results: list[SemanticSimilarityResult]
    mean_semantic_similarity: float
    mean_structural_similarity: float
    mean_semantic_gap: float
    gaming_detected: bool
    gaming_confidence: float      # [0, 1]
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("gaming_confidence")
    @classmethod
    def validate_gaming_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("gaming_confidence must be in [0, 1]")
        return v
