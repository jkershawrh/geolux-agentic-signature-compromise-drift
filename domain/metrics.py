from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator

from domain.enums import MetricDimension


def _new_id() -> str:
    return str(uuid.uuid4())


METRIC_DEFINITIONS: dict[MetricDimension, list[str]] = {
    MetricDimension.RESPONSE_STRUCTURE: [
        "avg_response_length",
        "response_length_variance",
        "paragraph_count",
        "code_block_ratio",
        "list_usage_frequency",
        "heading_depth",
    ],
    MetricDimension.TOKEN_ECONOMICS: [
        "input_output_ratio",
        "thinking_token_ratio",
        "cache_efficiency",
        "token_efficiency_score",
    ],
    MetricDimension.TOOL_BEHAVIOR: [
        "tool_call_frequency",
        "tool_sequence_entropy",
        "unique_tool_ratio",
        "tool_first_call_position",
        "tool_error_rate",
    ],
    MetricDimension.REASONING_PATTERN: [
        "thinking_engagement_rate",
        "thinking_depth",
        "step_count_distribution",
        "self_correction_frequency",
    ],
    MetricDimension.TEMPORAL_PROFILE: [
        "mean_latency_ms",
        "latency_variance",
        "time_to_first_token_ms",
        "latency_per_output_token",
    ],
    MetricDimension.SEMANTIC_CONSISTENCY: [
        "vocabulary_diversity",
        "sentiment_stability",
        "instruction_adherence_score",
    ],
    MetricDimension.SAFETY_ALIGNMENT: [
        "refusal_rate",
        "hedging_language_frequency",
        "boundary_testing_response",
    ],
    MetricDimension.AGENT_SPECIFIC: [
        "system_prompt_compliance",
        "response_signature_phrases",
        "closing_pattern",
    ],
}

ALL_METRIC_NAMES: list[str] = [
    name for names in METRIC_DEFINITIONS.values() for name in names
]

# All metrics are now implemented with text-analysis heuristics.
EXCLUDED_METRICS: set[str] = set()

PLACEHOLDER_METRICS: set[str] = set()


def get_exclusion_mask() -> list[bool]:
    """Return a mask — all True since every metric is now implemented."""
    return [True for _ in ALL_METRIC_NAMES]



class MetricMeasurement(BaseModel):
    metric_id: str = Field(default_factory=_new_id)
    run_id: str
    agent_id: str
    dimension: MetricDimension
    metric_name: str
    value: float
    normalized_value: float
    raw_data: dict[str, Any] | None = None

    @field_validator("normalized_value")
    @classmethod
    def validate_normalized(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("normalized_value must be in [0, 1]")
        return v

    @field_validator("metric_name")
    @classmethod
    def validate_metric_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("metric_name must not be empty")
        return v

    @field_validator("run_id", "agent_id")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field must not be empty")
        return v
