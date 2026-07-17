from __future__ import annotations

from typing import Optional

import numpy as np

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
from engine.geometric.distance import per_dimension_distances
from engine.geometric.embedding import metrics_to_vector
from engine.monitor import DriftMonitor
from engine.secret_beacon import SecretBeaconAuthenticator, StepUpResult
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
        secure: Optional[SecureMeasurement] = None,
        beacon_auth: Optional[SecretBeaconAuthenticator] = None,
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
        self._secure = secure or SecureMeasurement()
        self._beacons = beacon_auth or SecretBeaconAuthenticator()
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

        # Seal the baseline: encrypted copy + commitment hash, persisted so
        # later tampering with the plaintext baseline row is detectable via
        # verify_baseline_integrity().
        envelope = self._secure.encrypt_signature(baseline_signature)

        # Update agent status to ACTIVE
        if self._repo:
            self._repo.update_agent_status(agent.agent_id, AgentStatus.ACTIVE)
            self._repo.save_signature(baseline_signature)
            self._repo.save_envelope(envelope)

        return baseline_signature

    def verify_baseline_integrity(
        self, baseline: GeometricSignature
    ) -> tuple[bool, str]:
        """Check a stored baseline against its sealed envelope.

        Returns ``(ok, reason)``. Baselines with no recorded envelope
        (legacy, or saved outside assign()) pass with a note — sealing is
        opt-in evidence, not a gate on old data. Requires the same
        ASC_ENCRYPTION_KEY that sealed the envelope.
        """
        if not self._repo:
            return True, "no repository attached"
        envelope = self._repo.get_envelope_for_signature(baseline.signature_id)
        if envelope is None:
            return True, "no envelope recorded for this baseline"
        try:
            sealed_vector = self._secure.decrypt_signature(envelope)
        except ValueError as exc:
            return False, f"envelope verification failed: {exc}"
        if list(sealed_vector) != list(baseline.embedding_vector):
            return False, "stored baseline differs from sealed envelope (tampered)"
        return True, "verified against sealed envelope"

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

    def drift_breakdown(
        self,
        run: ControlledRun,
        baseline_signature: GeometricSignature,
    ) -> dict[str, float]:
        """Per-dimension drift decomposition for a single run vs. baseline.

        Answers "WHICH behavioral dimensions shifted", not just how much.
        """
        from engine.signature_generator import get_dimension_sizes

        metrics = self._extractor.extract(run)
        vec = metrics_to_vector(metrics)
        baseline_vec = np.array(baseline_signature.embedding_vector)
        return per_dimension_distances(vec, baseline_vec, get_dimension_sizes())

    # ------------------------------------------------------------------
    # Step-up verification (secret beacons)
    # ------------------------------------------------------------------

    def enroll_beacons(self, agent: AgentProfile, count: int = 6) -> str:
        """Provision secret beacons for an agent.

        Returns the covert system-prompt clause that must be deployed with
        the legitimate agent. Without the clause in its system prompt, the
        agent cannot pass step-up verification.
        """
        self._beacons.enroll(agent.agent_id, count)
        return self._beacons.system_prompt_clause(agent.agent_id)

    def step_up(self, agent: AgentProfile, n_challenges: int = 3) -> StepUpResult:
        """Run a secret-beacon challenge battery against the agent."""
        result = self._beacons.step_up(agent, self._adapter, n_challenges)
        if self._repo:
            self._repo.log_audit_event(
                "step_up",
                "passed" if result.passed else "failed",
                agent.agent_id,
            )
        return result

    def resolve_step_up(
        self,
        agent: AgentProfile,
        event: MonitoringEvent,
        policy: MonitoringPolicy,
        current_strike_count: int,
        result: StepUpResult,
    ) -> DriftAlert | None:
        """Convert a step-up outcome into an enforcement decision.

        Passed step-up: identity confirmed, drift logged without a strike
        (returns None). Failed step-up: a strike is applied and escalated
        through the graduated ladder.
        """
        alert = self._enforcement.resolve_step_up(
            event, policy, current_strike_count, result.passed
        )
        if alert and self._repo:
            self._repo.log_audit_event(
                "enforcement",
                alert.action_taken.value,
                agent.agent_id,
            )
        return alert

    @property
    def beacon_auth(self) -> SecretBeaconAuthenticator:
        return self._beacons

    def recertify(
        self,
        agent: AgentProfile,
        peer_baselines: list[GeometricSignature] | None = None,
    ) -> CertificationReport:
        """Re-run certification after changes."""
        return self.certify(agent, peer_baselines)
