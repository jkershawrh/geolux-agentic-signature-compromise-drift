from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from db.models import (
    AgentRow,
    AuditEventRow,
    CertificationRow,
    DriftMeasurementRow,
    MetricRow,
    MonitoringEventRow,
    ReducibilityRow,
    RunRow,
    SignatureEnvelopeRow,
    SignatureRow,
    StudyRow,
)
from domain.enums import AgentStatus
from domain.geometry import DriftMeasurement, GeometricSignature
from domain.identity import (
    CertificationReport,
    CertificationStatus,
    EnforcementAction,
    MonitoringEvent,
)
from domain.metrics import MetricMeasurement
from domain.models import AgentProfile, ControlledRun
from domain.reducibility import ReducibilityClassification
from domain.security import SecureSignatureEnvelope


class Repository:
    def __init__(self, session: Session):
        self._session = session

    # --- Agents ---

    def save_agent(self, agent: AgentProfile) -> None:
        row = AgentRow(
            agent_id=agent.agent_id,
            display_name=agent.display_name,
            model_id=agent.model_id,
            system_prompt_hash=agent.system_prompt_hash,
            tool_set_hash=agent.tool_set_hash,
            configuration=json.dumps(agent.configuration),
            status=agent.status.value,
            created_at=agent.created_at,
        )
        self._session.add(row)
        self._session.commit()

    def get_agent(self, agent_id: str) -> AgentProfile | None:
        row = self._session.query(AgentRow).filter_by(agent_id=agent_id).first()
        if not row:
            return None
        return AgentProfile(
            agent_id=row.agent_id,
            display_name=row.display_name,
            model_id=row.model_id,
            system_prompt_hash=row.system_prompt_hash,
            tool_set_hash=row.tool_set_hash,
            configuration=json.loads(row.configuration),
            status=AgentStatus(row.status),
            created_at=row.created_at,
        )

    def update_agent_status(self, agent_id: str, status: AgentStatus) -> None:
        row = self._session.query(AgentRow).filter_by(agent_id=agent_id).first()
        if row:
            row.status = status.value
            row.updated_at = datetime.now(timezone.utc)
            self._session.commit()

    def get_strike_count(self, agent_id: str) -> int:
        row = self._session.query(AgentRow).filter_by(agent_id=agent_id).first()
        return row.strike_count if row else 0

    def set_strike_count(self, agent_id: str, count: int) -> None:
        row = self._session.query(AgentRow).filter_by(agent_id=agent_id).first()
        if row:
            row.strike_count = count
            row.updated_at = datetime.now(timezone.utc)
            self._session.commit()

    def increment_strike_count(self, agent_id: str, delta: int = 1) -> int:
        """Atomically increment strike_count and return the new value."""
        from sqlalchemy import update

        stmt = (
            update(AgentRow)
            .where(AgentRow.agent_id == agent_id)
            .values(
                strike_count=AgentRow.strike_count + delta,
                updated_at=datetime.now(timezone.utc),
            )
        )
        self._session.execute(stmt)
        self._session.commit()
        row = self._session.query(AgentRow).filter_by(agent_id=agent_id).first()
        return row.strike_count if row else 0

    def list_agents(self) -> list[AgentProfile]:
        rows = self._session.query(AgentRow).all()
        return [
            AgentProfile(
                agent_id=r.agent_id,
                display_name=r.display_name,
                model_id=r.model_id,
                system_prompt_hash=r.system_prompt_hash,
                tool_set_hash=r.tool_set_hash,
                configuration=json.loads(r.configuration),
                status=AgentStatus(r.status),
                created_at=r.created_at,
            )
            for r in rows
        ]

    # --- Runs ---

    def save_run(self, run: ControlledRun) -> None:
        row = RunRow(
            run_id=run.run_id,
            agent_id=run.agent_id,
            scenario_id=run.scenario_id,
            prompt_hash=run.prompt_hash,
            prompt_text=run.prompt_text,
            response_text=run.response_text,
            model_id=run.model_id,
            input_tokens=run.input_tokens,
            output_tokens=run.output_tokens,
            cache_read_tokens=run.cache_read_tokens,
            cache_creation_tokens=run.cache_creation_tokens,
            latency_ms=run.latency_ms,
            time_to_first_token_ms=run.time_to_first_token_ms,
            stop_reason=run.stop_reason,
            thinking_tokens=run.thinking_tokens,
            tool_calls_json=json.dumps(run.tool_calls),
            tool_call_count=run.tool_call_count,
            tool_sequence_json=json.dumps(run.tool_sequence),
            raw_usage_json=json.dumps(run.raw_usage),
            perturbation_json=json.dumps(run.perturbation_applied) if run.perturbation_applied else None,
            status=run.status.value,
            created_at=run.created_at,
        )
        self._session.add(row)
        self._session.commit()

    def get_runs_for_agent(self, agent_id: str) -> list[ControlledRun]:
        rows = self._session.query(RunRow).filter_by(agent_id=agent_id).all()
        return [self._row_to_run(r) for r in rows]

    def _row_to_run(self, r: RunRow) -> ControlledRun:
        return ControlledRun(
            run_id=r.run_id,
            agent_id=r.agent_id,
            scenario_id=r.scenario_id,
            prompt_hash=r.prompt_hash,
            prompt_text=r.prompt_text,
            response_text=r.response_text,
            model_id=r.model_id,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            cache_read_tokens=r.cache_read_tokens,
            cache_creation_tokens=r.cache_creation_tokens,
            latency_ms=r.latency_ms,
            time_to_first_token_ms=r.time_to_first_token_ms,
            stop_reason=r.stop_reason,
            thinking_tokens=r.thinking_tokens,
            tool_calls=json.loads(r.tool_calls_json),
            tool_call_count=r.tool_call_count,
            tool_sequence=json.loads(r.tool_sequence_json),
            raw_usage=json.loads(r.raw_usage_json),
            perturbation_applied=json.loads(r.perturbation_json) if r.perturbation_json else None,
            status=r.status,
            created_at=r.created_at,
        )

    # --- Metrics ---

    def save_metrics(self, metrics: list[MetricMeasurement]) -> None:
        for m in metrics:
            row = MetricRow(
                metric_id=m.metric_id,
                run_id=m.run_id,
                agent_id=m.agent_id,
                dimension=m.dimension.value,
                metric_name=m.metric_name,
                value=m.value,
                normalized_value=m.normalized_value,
                raw_data_json=json.dumps(m.raw_data) if m.raw_data else None,
            )
            self._session.add(row)
        self._session.commit()

    def get_metrics_for_agent(self, agent_id: str) -> list[MetricMeasurement]:
        rows = self._session.query(MetricRow).filter_by(agent_id=agent_id).all()
        return [
            MetricMeasurement(
                metric_id=r.metric_id,
                run_id=r.run_id,
                agent_id=r.agent_id,
                dimension=r.dimension,
                metric_name=r.metric_name,
                value=r.value,
                normalized_value=r.normalized_value,
                raw_data=json.loads(r.raw_data_json) if r.raw_data_json else None,
            )
            for r in rows
        ]

    # --- Signatures ---

    def save_signature(self, sig: GeometricSignature) -> None:
        row = SignatureRow(
            signature_id=sig.signature_id,
            agent_id=sig.agent_id,
            signature_type=sig.signature_type.value,
            embedding_vector_json=json.dumps(sig.embedding_vector),
            embedding_dimension=sig.embedding_dimension,
            manifold_coordinates_json=json.dumps(sig.manifold_coordinates),
            metric_tensor_json=json.dumps(sig.metric_tensor) if sig.metric_tensor else None,
            metric_snapshot_json=json.dumps(sig.metric_snapshot),
            run_ids_json=json.dumps(sig.run_ids),
            num_runs=sig.num_runs,
            computation_method=sig.computation_method,
            stability_score=sig.stability_score,
            created_at=sig.created_at,
        )
        self._session.add(row)
        self._session.commit()

    def save_envelope(self, envelope: SecureSignatureEnvelope) -> None:
        row = SignatureEnvelopeRow(
            envelope_id=envelope.envelope_id,
            agent_id=envelope.agent_id,
            signature_id=envelope.signature_id,
            encrypted_vector=envelope.encrypted_vector,
            commitment_hash=envelope.commitment_hash,
            created_at=envelope.created_at,
        )
        self._session.add(row)
        self._session.commit()

    def get_envelope_for_signature(self, signature_id: str) -> SecureSignatureEnvelope | None:
        row = (
            self._session.query(SignatureEnvelopeRow)
            .filter_by(signature_id=signature_id)
            .first()
        )
        if not row:
            return None
        return SecureSignatureEnvelope(
            envelope_id=row.envelope_id,
            agent_id=row.agent_id,
            signature_id=row.signature_id,
            encrypted_vector=row.encrypted_vector,
            commitment_hash=row.commitment_hash,
            created_at=row.created_at,
        )

    def get_baseline_signature(self, agent_id: str) -> GeometricSignature | None:
        row = (
            self._session.query(SignatureRow)
            .filter_by(agent_id=agent_id, signature_type="baseline")
            .order_by(SignatureRow.created_at.desc())
            .first()
        )
        if not row:
            return None
        return self._row_to_signature(row)

    def _row_to_signature(self, r: SignatureRow) -> GeometricSignature:
        return GeometricSignature(
            signature_id=r.signature_id,
            agent_id=r.agent_id,
            signature_type=r.signature_type,
            embedding_vector=json.loads(r.embedding_vector_json),
            embedding_dimension=r.embedding_dimension,
            manifold_coordinates=json.loads(r.manifold_coordinates_json),
            metric_tensor=json.loads(r.metric_tensor_json) if r.metric_tensor_json else None,
            metric_snapshot=json.loads(r.metric_snapshot_json),
            run_ids=json.loads(r.run_ids_json),
            num_runs=r.num_runs,
            computation_method=r.computation_method,
            stability_score=r.stability_score,
            created_at=r.created_at,
        )

    # --- Drift Measurements ---

    def save_drift_measurement(self, drift: DriftMeasurement) -> None:
        row = DriftMeasurementRow(
            measurement_id=drift.measurement_id,
            agent_id=drift.agent_id,
            baseline_signature_id=drift.baseline_signature_id,
            current_signature_id=drift.current_signature_id,
            geodesic_distance=drift.geodesic_distance,
            euclidean_distance=drift.euclidean_distance,
            cosine_similarity=drift.cosine_similarity,
            drift_category=drift.drift_category.value,
            drift_magnitude=drift.drift_magnitude,
            drift_direction_json=json.dumps(drift.drift_direction) if drift.drift_direction else None,
            per_dimension_drift_json=json.dumps(drift.per_dimension_drift),
            is_significant=drift.is_significant,
            p_value=drift.p_value,
            compromise_probability=drift.compromise_probability,
            created_at=drift.created_at,
        )
        self._session.add(row)
        self._session.commit()

    # --- Reducibility Classifications ---

    def save_reducibility(self, classification: ReducibilityClassification) -> None:
        row = ReducibilityRow(
            classification_id=classification.classification_id,
            agent_id=classification.agent_id,
            dimension=classification.dimension.value,
            metric_name=classification.metric_name,
            reducibility=classification.reducibility.value,
            predictability_score=classification.predictability_score,
            variance=classification.variance,
            evidence_json=json.dumps(classification.evidence),
            sample_size=classification.sample_size,
            created_at=classification.created_at,
        )
        self._session.add(row)
        self._session.commit()

    # --- Certifications ---

    def save_certification(self, report: CertificationReport) -> None:
        row = CertificationRow(
            certification_id=report.certification_id,
            agent_id=report.agent_id,
            status=report.status.value,
            self_consistency_json=json.dumps(report.self_consistency_distances),
            discriminability_json=json.dumps(report.discriminability_scores),
            canary_pass_rate=report.canary_pass_rate,
            multi_turn_scores_json=json.dumps(report.multi_turn_scores),
            attack_detection_rate=report.attack_detection_rate,
            all_passed=report.all_checks_passed,
            failure_reasons_json=json.dumps(report.failure_reasons),
            report_json=json.dumps(report.model_dump(), default=str),
            created_at=report.created_at,
        )
        self._session.add(row)
        self._session.commit()

    def get_certifications_for_agent(self, agent_id: str) -> list[CertificationReport]:
        rows = (
            self._session.query(CertificationRow)
            .filter_by(agent_id=agent_id)
            .order_by(CertificationRow.created_at.desc())
            .all()
        )
        return [self._row_to_certification(r) for r in rows]

    def _row_to_certification(self, r: CertificationRow) -> CertificationReport:
        return CertificationReport(
            certification_id=r.certification_id,
            agent_id=r.agent_id,
            status=CertificationStatus(r.status),
            self_consistency_distances=json.loads(r.self_consistency_json),
            self_consistency_passed=json.loads(r.report_json).get("self_consistency_passed", False),
            discriminability_scores=json.loads(r.discriminability_json),
            discriminability_passed=json.loads(r.report_json).get("discriminability_passed", False),
            canary_pass_rate=r.canary_pass_rate,
            canary_passed=json.loads(r.report_json).get("canary_passed", False),
            multi_turn_scores=json.loads(r.multi_turn_scores_json),
            multi_turn_passed=json.loads(r.report_json).get("multi_turn_passed", False),
            attack_detection_rate=r.attack_detection_rate,
            attack_passed=json.loads(r.report_json).get("attack_passed", False),
            all_checks_passed=r.all_passed,
            failure_reasons=json.loads(r.failure_reasons_json),
            created_at=r.created_at,
        )

    # --- Monitoring Events ---

    def save_monitoring_event(self, event: MonitoringEvent) -> None:
        row = MonitoringEventRow(
            event_id=event.event_id,
            agent_id=event.agent_id,
            event_type=event.event_type,
            drift_score=event.drift_score,
            threshold_used=event.threshold_used,
            action_taken=event.action_taken.value,
            created_at=event.created_at,
        )
        self._session.add(row)
        self._session.commit()

    def get_monitoring_events(self, agent_id: str, limit: int = 50) -> list[MonitoringEvent]:
        rows = (
            self._session.query(MonitoringEventRow)
            .filter_by(agent_id=agent_id)
            .order_by(MonitoringEventRow.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            MonitoringEvent(
                event_id=r.event_id,
                agent_id=r.agent_id,
                event_type=r.event_type,
                drift_score=r.drift_score,
                threshold_used=r.threshold_used,
                action_taken=EnforcementAction(r.action_taken),
                created_at=r.created_at,
            )
            for r in rows
        ]

    # --- Agent Strike Count ---

    def update_agent_strike_count(self, agent_id: str, count: int) -> None:
        row = self._session.query(AgentRow).filter_by(agent_id=agent_id).first()
        if row:
            row.strike_count = count
            row.updated_at = datetime.now(timezone.utc)
            self._session.commit()

    # --- Studies ---

    def save_study(self, study_id: str, study_name: str, model_id: str,
                   agents_count: int, runs_per_agent: int) -> None:
        row = StudyRow(
            study_id=study_id,
            study_name=study_name,
            model_id=model_id,
            agents_count=agents_count,
            runs_per_agent=runs_per_agent,
            status="running",
            started_at=datetime.now(timezone.utc),
        )
        self._session.add(row)
        self._session.commit()

    def complete_study(self, study_id: str, results: dict) -> None:
        row = self._session.query(StudyRow).filter_by(study_id=study_id).first()
        if row:
            row.status = "completed"
            row.completed_at = datetime.now(timezone.utc)
            row.results_json = json.dumps(results)
            row.total_runs = results.get("total_runs", 0)
            self._session.commit()

    def fail_study(self, study_id: str, error: str) -> None:
        row = self._session.query(StudyRow).filter_by(study_id=study_id).first()
        if row:
            row.status = "failed"
            row.completed_at = datetime.now(timezone.utc)
            row.results_json = json.dumps({"error": error})
            self._session.commit()

    def list_studies(self) -> list:
        return self._session.query(StudyRow).order_by(StudyRow.created_at).all()

    # --- Audit Events ---

    def log_audit_event(
        self,
        source_component: str,
        event_type: str,
        agent_id: str | None = None,
        payload: dict | None = None,
    ) -> None:
        last_event = (
            self._session.query(AuditEventRow)
            .order_by(AuditEventRow.id.desc())
            .first()
        )
        previous_hash = last_event.event_hash if last_event else "0" * 64

        event_data = f"{source_component}:{event_type}:{agent_id}:{json.dumps(payload)}:{previous_hash}"
        event_hash = hashlib.sha256(event_data.encode()).hexdigest()

        row = AuditEventRow(
            event_id=str(uuid.uuid4()),
            source_component=source_component,
            event_type=event_type,
            agent_id=agent_id,
            payload_json=json.dumps(payload) if payload else None,
            previous_hash=previous_hash,
            event_hash=event_hash,
        )
        self._session.add(row)
        self._session.commit()
