from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentRow(Base):
    __tablename__ = "asc_agents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    system_prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    tool_set_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    configuration: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="baseline_pending")
    monitoring_policy: Mapped[str] = mapped_column(String(50), nullable=False, default="graduated")
    monitoring_frequency: Mapped[str] = mapped_column(String(50), nullable=False, default="adaptive")
    strike_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class RunRow(Base):
    __tablename__ = "asc_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    agent_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    scenario_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    response_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_read_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_creation_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    time_to_first_token_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stop_reason: Mapped[str] = mapped_column(String(50), nullable=False, default="end_turn")
    thinking_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_calls_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    tool_call_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_sequence_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    raw_usage_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    perturbation_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    study_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class MetricRow(Base):
    __tablename__ = "asc_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    metric_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    agent_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    dimension: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metric_name: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    normalized_value: Mapped[float] = mapped_column(Float, nullable=False)
    raw_data_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        Index("ix_metrics_agent_dim_time", "agent_id", "dimension", "created_at"),
    )


class SignatureRow(Base):
    __tablename__ = "asc_signatures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signature_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    agent_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    signature_type: Mapped[str] = mapped_column(String(50), nullable=False)
    embedding_vector_json: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    manifold_coordinates_json: Mapped[str] = mapped_column(Text, nullable=False)
    metric_tensor_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metric_snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)
    run_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    num_runs: Mapped[int] = mapped_column(Integer, nullable=False)
    computation_method: Mapped[str] = mapped_column(String(100), nullable=False)
    stability_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        Index("ix_sig_agent_type_time", "agent_id", "signature_type", "created_at"),
    )


class DriftMeasurementRow(Base):
    __tablename__ = "asc_drift_measurements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    measurement_id: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    agent_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    baseline_signature_id: Mapped[str] = mapped_column(String(255), nullable=False)
    current_signature_id: Mapped[str] = mapped_column(String(255), nullable=False)
    geodesic_distance: Mapped[float] = mapped_column(Float, nullable=False)
    euclidean_distance: Mapped[float] = mapped_column(Float, nullable=False)
    cosine_similarity: Mapped[float] = mapped_column(Float, nullable=False)
    drift_category: Mapped[str] = mapped_column(String(50), nullable=False)
    drift_magnitude: Mapped[float] = mapped_column(Float, nullable=False)
    drift_direction_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    per_dimension_drift_json: Mapped[str] = mapped_column(Text, nullable=False)
    is_significant: Mapped[bool] = mapped_column(Boolean, nullable=False)
    p_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    compromise_probability: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ReducibilityRow(Base):
    __tablename__ = "asc_reducibility_classifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    classification_id: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    agent_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    dimension: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(255), nullable=False)
    reducibility: Mapped[str] = mapped_column(String(50), nullable=False)
    predictability_score: Mapped[float] = mapped_column(Float, nullable=False)
    variance: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class RubricScoreRow(Base):
    __tablename__ = "asc_rubric_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    score_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    rubric_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    dimension: Mapped[str] = mapped_column(String(100), nullable=False)
    state: Mapped[str] = mapped_column(String(10), nullable=False)
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    test_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class CertificationRow(Base):
    __tablename__ = "asc_certifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    certification_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    agent_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    self_consistency_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    discriminability_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    canary_pass_rate: Mapped[float] = mapped_column(Float, nullable=False)
    multi_turn_scores_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    attack_detection_rate: Mapped[float] = mapped_column(Float, nullable=False)
    all_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    failure_reasons_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    report_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class MonitoringEventRow(Base):
    __tablename__ = "asc_monitoring_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    agent_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    drift_score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_used: Mapped[float] = mapped_column(Float, nullable=False)
    action_taken: Mapped[str] = mapped_column(String(50), nullable=False, default="none")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AuditEventRow(Base):
    __tablename__ = "asc_audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    source_component: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    agent_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    payload_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    previous_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class StudyRow(Base):
    __tablename__ = "asc_studies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    study_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    study_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    agents_count: Mapped[int] = mapped_column(Integer, nullable=False)
    runs_per_agent: Mapped[int] = mapped_column(Integer, nullable=False)
    total_runs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    results_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
