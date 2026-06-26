from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _new_id() -> str:
    return str(uuid.uuid4())


class ProbeCategory(str, Enum):
    FACTUAL_RECALL = "factual_recall"
    REASONING_CHAIN = "reasoning_chain"
    STYLE_COMPLIANCE = "style_compliance"
    BOUNDARY_PROBE = "boundary_probe"
    CANARY_PROBE = "canary_probe"


class ProbeTemplate(BaseModel):
    template_id: str = Field(default_factory=_new_id)
    category: ProbeCategory
    template_text: str
    substitution_pool: dict[str, list[str]]
    difficulty: float
    expected_properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("difficulty must be in [0, 1]")
        return v

    @field_validator("template_text")
    @classmethod
    def validate_template_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("template_text must not be empty")
        return v


class GeneratedProbe(BaseModel):
    probe_id: str = Field(default_factory=_new_id)
    template_id: str
    category: ProbeCategory
    prompt_text: str
    prompt_hash: str
    substitutions_used: dict[str, str]
    difficulty: float
    expected_properties: dict[str, Any] = Field(default_factory=dict)
    seed: int

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("difficulty must be in [0, 1]")
        return v

    @field_validator("prompt_text")
    @classmethod
    def validate_prompt_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt_text must not be empty")
        return v


class ProbeSet(BaseModel):
    probe_set_id: str = Field(default_factory=_new_id)
    probes: list[GeneratedProbe]
    category_distribution: dict[str, int]
    total_count: int
    generation_seed: int

    @field_validator("total_count")
    @classmethod
    def validate_total_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError("total_count must be non-negative")
        return v
