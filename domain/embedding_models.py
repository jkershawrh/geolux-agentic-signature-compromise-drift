from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class EmbeddingBaseline(BaseModel):
    baseline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    centroid: list[float]
    threshold: float
    within_mean: float
    within_std: float
    pca_components: list[list[float]]
    pca_mean: list[float]
    explained_variance: float
    n_components: int
    n_responses: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
