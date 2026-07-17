from __future__ import annotations

from domain.identity import (
    DriftAlert,
    EnforcementAction,
    MonitoringEvent,
    MonitoringPolicy,
)


class EnforcementEngine:
    """Response policy engine for drift enforcement.

    Evaluates monitoring events against the configured policy and
    strike count to determine the appropriate enforcement action.

    Supports four policies:
    - ALERT_ONLY: always issue a warning, never escalate
    - KILL_SWITCH: immediately suspend on any drift
    - GRADUATED: escalation ladder (warning -> throttle -> suspend)
    - GRADUATED_STEP_UP: challenge-first — drift returns a STEP_UP alert
      demanding secret-beacon re-verification; strikes are applied via
      ``resolve_step_up`` only when the verification FAILS. A drift alarm
      alone (which may be a statistical false positive) never costs a
      strike under this policy.
    """

    def __init__(
        self,
        cooldown_period_seconds: int = 3600,
        max_strikes: int = 3,
    ):
        self._cooldown_period_seconds = cooldown_period_seconds
        self._max_strikes = max_strikes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        monitoring_event: MonitoringEvent,
        policy: MonitoringPolicy,
        current_strike_count: int,
    ) -> DriftAlert | None:
        """Determine enforcement action based on policy and strike count.

        Returns None if no drift was detected (action == NONE).
        """
        if monitoring_event.action_taken == EnforcementAction.NONE:
            return None  # No drift detected

        if policy == MonitoringPolicy.ALERT_ONLY:
            return DriftAlert(
                agent_id=monitoring_event.agent_id,
                severity="warning",
                drift_score=monitoring_event.drift_score,
                threshold=monitoring_event.threshold_used,
                action_taken=EnforcementAction.WARNING,
                strike_count=current_strike_count,
                details={
                    "policy": policy.value,
                    "event_type": monitoring_event.event_type,
                },
            )

        if policy == MonitoringPolicy.KILL_SWITCH:
            return DriftAlert(
                agent_id=monitoring_event.agent_id,
                severity="suspend",
                drift_score=monitoring_event.drift_score,
                threshold=monitoring_event.threshold_used,
                action_taken=EnforcementAction.SUSPEND,
                strike_count=current_strike_count,
                details={
                    "policy": policy.value,
                    "event_type": monitoring_event.event_type,
                    "reason": "kill_switch_triggered",
                },
            )

        if policy == MonitoringPolicy.GRADUATED_STEP_UP:
            # Challenge first: no strike until the step-up verdict is known.
            return DriftAlert(
                agent_id=monitoring_event.agent_id,
                severity="challenge",
                drift_score=monitoring_event.drift_score,
                threshold=monitoring_event.threshold_used,
                action_taken=EnforcementAction.STEP_UP,
                strike_count=current_strike_count,
                details={
                    "policy": policy.value,
                    "event_type": monitoring_event.event_type,
                    "reason": "drift_detected_pending_step_up",
                },
            )

        # GRADUATED: escalation ladder
        return self._graduated_ladder(
            monitoring_event, policy, current_strike_count + 1
        )

    def resolve_step_up(
        self,
        monitoring_event: MonitoringEvent,
        policy: MonitoringPolicy,
        current_strike_count: int,
        step_up_passed: bool,
    ) -> DriftAlert | None:
        """Apply the outcome of a step-up verification.

        A passed step-up confirms identity: the drift stands as an
        observation but costs no strike (returns None). A failed step-up
        is treated as a strike and escalated through the graduated ladder.
        """
        if step_up_passed:
            return None
        alert = self._graduated_ladder(
            monitoring_event, policy, current_strike_count + 1
        )
        alert.details["reason"] = (
            f"step_up_failed ({alert.details['reason']})"
        )
        return alert

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _graduated_ladder(
        self,
        monitoring_event: MonitoringEvent,
        policy: MonitoringPolicy,
        new_strikes: int,
    ) -> DriftAlert:
        """Warning -> throttle -> suspend based on the new strike count."""
        if new_strikes >= self._max_strikes:
            severity, action = "suspend", EnforcementAction.SUSPEND
            reason = f"max_strikes_reached ({new_strikes}/{self._max_strikes})"
        elif new_strikes >= 2:
            severity, action = "throttle", EnforcementAction.THROTTLE
            reason = f"strike_{new_strikes}_of_{self._max_strikes}"
        else:
            severity, action = "warning", EnforcementAction.WARNING
            reason = f"strike_{new_strikes}_of_{self._max_strikes}"

        return DriftAlert(
            agent_id=monitoring_event.agent_id,
            severity=severity,
            drift_score=monitoring_event.drift_score,
            threshold=monitoring_event.threshold_used,
            action_taken=action,
            strike_count=new_strikes,
            details={
                "policy": policy.value,
                "event_type": monitoring_event.event_type,
                "reason": reason,
            },
        )
