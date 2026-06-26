from __future__ import annotations

from typing import Optional

from adapters.interfaces import InferenceAdapter, MetricExtractor
from domain.enums import AgentStatus
from domain.geometry import GeometricSignature
from domain.identity import (
    CertificationReport,
    DriftAlert,
    EnrollmentRequest,
    MonitoringEvent,
    MonitoringPolicy,
)
from domain.models import AgentProfile, ControlledRun
from engine.certification import CertificationEngine
from engine.drift_detector import DriftDetector
from engine.enforcement import EnforcementEngine
from engine.monitor import DriftMonitor
from engine.secure_measurement import SecureMeasurement
from engine.signature_generator import SignatureGenerator


class IdentityPipeline:
    """Main orchestrator for the Agent Identity Pipeline.

    Composes certification, monitoring, enforcement, and secure
    measurement into a single lifecycle:

        enroll -> certify -> assign -> monitor -> respond -> recertify
    """

    def __init__(
        self,
        adapter: InferenceAdapter,
        extractor: MetricExtractor,
        generator: Optional[SignatureGenerator] = None,
        repository: Optional[object] = None,
        canary_threshold: float = 0.8,
        attack_threshold: float = 0.7,
    ):
        self._adapter = adapter
        self._extractor = extractor
        self._generator = generator or SignatureGenerator(manifold_method="pca")
        self._repo = repository
        self._certification = CertificationEngine(
            adapter, extractor, self._generator,
            canary_threshold=canary_threshold,
            attack_threshold=attack_threshold,
        )
        self._monitor = DriftMonitor(extractor, self._generator, DriftDetector())
        self._enforcement = EnforcementEngine()
        self._secure = SecureMeasurement()
        self._discriminative_masks: dict[str, list[bool]] = {}

    # ------------------------------------------------------------------
    # Lifecycle stages
    # ------------------------------------------------------------------

    def enroll(self, request: EnrollmentRequest) -> AgentProfile:
        """Create agent from enrollment request. Status = ENROLLED."""
        agent = AgentProfile(
            agent_id=request.agent_id,
            display_name=request.display_name,
            model_id=request.model_id,
            system_prompt=request.system_prompt,
            status=AgentStatus.ENROLLED,
        )
        if self._repo:
            self._repo.save_agent(agent)
        return agent

    def certify(
        self,
        agent: AgentProfile,
        peer_baselines: list[GeometricSignature] | None = None,
    ) -> CertificationReport:
        """Run certification battery. Status -> CERTIFIED or stays ENROLLED."""
        report = self._certification.certify(agent, peer_baselines)
        if report.all_checks_passed and self._repo:
            self._repo.update_agent_status(agent.agent_id, AgentStatus.CERTIFIED)
        if self._repo:
            self._repo.save_certification(report)
        return report

    def assign(
        self,
        agent: AgentProfile,
        certification_report: CertificationReport,
        baseline_signature: GeometricSignature,
    ) -> GeometricSignature | None:
        """If certified, encrypt and lock the signature. Status -> ACTIVE."""
        if not certification_report.all_checks_passed:
            return None

        # Store the discriminative mask for later monitoring
        self._discriminative_masks[agent.agent_id] = (
            certification_report.discriminative_mask or []
        )

        # Encrypt the baseline signature
        envelope = self._secure.encrypt_signature(baseline_signature)

        # Update agent status to ACTIVE
        if self._repo:
            self._repo.update_agent_status(agent.agent_id, AgentStatus.ACTIVE)
            self._repo.save_signature(baseline_signature)

        return baseline_signature

    def monitor(
        self,
        agent: AgentProfile,
        run: ControlledRun,
        baseline_signature: GeometricSignature,
        metric_mask: list[bool] | None = None,
    ) -> MonitoringEvent:
        """Check a single run against baseline.

        If *metric_mask* is not provided, the mask stored during ``assign()``
        is used automatically (if one exists for this agent).
        """
        if metric_mask is None:
            metric_mask = self._discriminative_masks.get(agent.agent_id) or None
        event = self._monitor.inline_check(run, baseline_signature, metric_mask)
        if self._repo:
            self._repo.save_monitoring_event(event)
        return event

    def respond(
        self,
        agent: AgentProfile,
        event: MonitoringEvent,
        policy: MonitoringPolicy,
        current_strike_count: int = 0,
    ) -> DriftAlert | None:
        """Evaluate enforcement based on policy."""
        alert = self._enforcement.evaluate(event, policy, current_strike_count)
        if alert and self._repo:
            self._repo.log_audit_event(
                "enforcement",
                alert.action_taken.value,
                agent.agent_id,
            )
        return alert

    def recertify(
        self,
        agent: AgentProfile,
        peer_baselines: list[GeometricSignature] | None = None,
    ) -> CertificationReport:
        """Re-run certification after changes."""
        return self.certify(agent, peer_baselines)
