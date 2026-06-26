from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator, model_validator

from domain.enums import DriftCategory, SignatureType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class GeometricSignature(BaseModel):
    signature_id: str = Field(default_factory=_new_id)
    agent_id: str
    signature_type: SignatureType
    embedding_vector: list[float]
    embedding_dimension: int
    manifold_coordinates: list[float]
    metric_tensor: list[list[float]] | None = None
    metric_snapshot: dict[str, float]
    run_ids: list[str]
    num_runs: int
    computation_method: str
    stability_score: float | None = None
    per_run_vectors: list[list[float]] | None = None
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("embedding_vector")
    @classmethod
    def embedding_not_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("embedding_vector must not be empty")
        return v

    @field_validator("run_ids")
    @classmethod
    def run_ids_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("run_ids must not be empty")
        return v

    @field_validator("num_runs")
    @classmethod
    def num_runs_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("num_runs must be positive")
        return v

    @field_validator("stability_score")
    @classmethod
    def stability_in_range(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("stability_score must be in [0, 1]")
        return v

    @model_validator(mode="after")
    def dimension_matches_vector(self) -> "GeometricSignature":
        if self.embedding_dimension != len(self.embedding_vector):
            raise ValueError(
                f"embedding_dimension ({self.embedding_dimension}) "
                f"must match len(embedding_vector) ({len(self.embedding_vector)})"
            )
        return self


class DriftMeasurement(BaseModel):
    measurement_id: str = Field(default_factory=_new_id)
    agent_id: str
    baseline_signature_id: str
    current_signature_id: str
    geodesic_distance: float
    euclidean_distance: float
    cosine_similarity: float
    drift_category: DriftCategory
    drift_magnitude: float
    drift_direction: list[float] | None = None
    per_dimension_drift: dict[str, float]
    is_significant: bool
    p_value: float | None = None
    compromise_probability: float
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("geodesic_distance", "euclidean_distance")
    @classmethod
    def distance_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("distance must be non-negative")
        return v

    @field_validator("cosine_similarity")
    @classmethod
    def cosine_in_range(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError("cosine_similarity must be in [-1, 1]")
        return v

    @field_validator("drift_magnitude")
    @classmethod
    def magnitude_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("drift_magnitude must be in [0, 1]")
        return v

    @field_validator("compromise_probability")
    @classmethod
    def probability_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("compromise_probability must be in [0, 1]")
        return v

    @field_validator("p_value")
    @classmethod
    def p_value_in_range(cls, v: float | None) -> float | None:
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("p_value must be in [0, 1]")
        return v
