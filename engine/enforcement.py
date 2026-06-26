from __future__ import annotations

from typing import Optional

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

    Supports three policies:
    - ALERT_ONLY: always issue a warning, never escalate
    - KILL_SWITCH: immediately suspend on any drift
    - GRADUATED: escalation ladder (warning -> throttle -> suspend)
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

        # GRADUATED: escalation ladder
        new_strikes = current_strike_count + 1

        if new_strikes >= self._max_strikes:
            return DriftAlert(
                agent_id=monitoring_event.agent_id,
                severity="suspend",
                drift_score=monitoring_event.drift_score,
                threshold=monitoring_event.threshold_used,
                action_taken=EnforcementAction.SUSPEND,
                strike_count=new_strikes,
                details={
                    "policy": policy.value,
                    "event_type": monitoring_event.event_type,
                    "reason": f"max_strikes_reached ({new_strikes}/{self._max_strikes})",
                },
            )
        elif new_strikes >= 2:
            return DriftAlert(
                agent_id=monitoring_event.agent_id,
                severity="throttle",
                drift_score=monitoring_event.drift_score,
                threshold=monitoring_event.threshold_used,
                action_taken=EnforcementAction.THROTTLE,
                strike_count=new_strikes,
                details={
                    "policy": policy.value,
                    "event_type": monitoring_event.event_type,
                    "reason": f"strike_{new_strikes}_of_{self._max_strikes}",
                },
            )
        else:
            return DriftAlert(
                agent_id=monitoring_event.agent_id,
                severity="warning",
                drift_score=monitoring_event.drift_score,
                threshold=monitoring_event.threshold_used,
                action_taken=EnforcementAction.WARNING,
                strike_count=new_strikes,
                details={
                    "policy": policy.value,
                    "event_type": monitoring_event.event_type,
                    "reason": f"strike_{new_strikes}_of_{self._max_strikes}",
                },
            )
