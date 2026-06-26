from __future__ import annotations

from typing import Optional

import numpy as np

from adapters.interfaces import MetricExtractor
from domain.geometry import GeometricSignature
from domain.identity import EnforcementAction, MonitoringEvent, MonitoringFrequency
from domain.metrics import MetricMeasurement
from domain.models import ControlledRun
from engine.drift_detector import DriftDetector
from engine.geometric.distance import euclidean_distance
from engine.geometric.embedding import metrics_to_vector
from engine.signature_generator import SignatureGenerator


class DriftMonitor:
    """Continuous monitoring engine for agent identity drift.

    Provides two modes of checking:
    - ``inline_check``: lightweight per-call Euclidean distance check
    - ``periodic_check``: full signature generation + drift detection

    Also provides adaptive frequency escalation based on recent events.
    """

    def __init__(
        self,
        extractor: MetricExtractor,
        generator: SignatureGenerator,
        drift_detector: DriftDetector,
        inline_threshold: float = 0.3,
        periodic_threshold: float = 0.2,
    ):
        self._extractor = extractor
        self._generator = generator
        self._drift_detector = drift_detector
        self._inline_threshold = inline_threshold
        self._periodic_threshold = periodic_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inline_check(
        self,
        run: ControlledRun,
        baseline_signature: GeometricSignature,
        metric_mask: list[bool] | None = None,
    ) -> MonitoringEvent:
        """Lightweight per-call check.

        Extract metrics from a single run, compute Euclidean distance
        from baseline centroid, and return a monitoring event.

        If *metric_mask* is provided, zero out non-selected dimensions
        before computing distance so only discriminative metrics contribute.
        """
        metrics = self._extractor.extract(run)
        vec = metrics_to_vector(metrics)
        baseline_vec = np.array(baseline_signature.embedding_vector)

        if metric_mask is not None:
            mask_arr = np.array([1.0 if m else 0.0 for m in metric_mask])
            vec = vec * mask_arr
            baseline_vec = baseline_vec * mask_arr

        dist = float(euclidean_distance(vec, baseline_vec))

        action = EnforcementAction.NONE
        if dist > self._inline_threshold:
            action = EnforcementAction.WARNING

        return MonitoringEvent(
            agent_id=run.agent_id,
            event_type="inline_check",
            drift_score=dist,
            threshold_used=self._inline_threshold,
            action_taken=action,
        )

    def periodic_check(
        self,
        runs: list[ControlledRun],
        baseline_signature: GeometricSignature,
    ) -> MonitoringEvent:
        """Deep batch check.

        Compute a full signature from accumulated runs, run drift
        detector against the baseline, and return a monitoring event
        with drift magnitude.
        """
        if not runs:
            return MonitoringEvent(
                agent_id="unknown",
                event_type="periodic_check",
                drift_score=0.0,
                threshold_used=self._periodic_threshold,
                action_taken=EnforcementAction.NONE,
            )

        agent_id = runs[0].agent_id

        # Extract metrics for all runs
        all_metrics: list[list[MetricMeasurement]] = []
        run_ids: list[str] = []
        for run in runs:
            metrics = self._extractor.extract(run)
            all_metrics.append(metrics)
            run_ids.append(run.run_id)

        # Generate current signature
        current_sig = self._generator.generate(
            agent_id=agent_id,
            metrics_per_run=all_metrics,
            run_ids=run_ids,
        )

        # Run drift detection
        drift = self._drift_detector.detect(baseline_signature, current_sig)

        action = EnforcementAction.NONE
        if drift.drift_magnitude > self._periodic_threshold:
            action = EnforcementAction.WARNING

        return MonitoringEvent(
            agent_id=agent_id,
            event_type="periodic_check",
            drift_score=drift.drift_magnitude,
            threshold_used=self._periodic_threshold,
            action_taken=action,
        )

    def should_escalate(
        self,
        frequency: MonitoringFrequency,
        recent_events: list[MonitoringEvent],
    ) -> MonitoringFrequency:
        """Adaptive monitoring: if recent drift velocity is increasing,
        recommend switching from periodic to inline.
        """
        if frequency == MonitoringFrequency.ADAPTIVE:
            warning_count = sum(
                1 for e in recent_events
                if e.action_taken != EnforcementAction.NONE
            )
            if warning_count >= 2:
                return MonitoringFrequency.INLINE
            return MonitoringFrequency.PERIODIC_5M
        return frequency
