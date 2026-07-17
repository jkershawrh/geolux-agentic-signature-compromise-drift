"""Secret-beacon step-up verification: authenticator, enforcement, pipeline."""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adapters.metric_extractor import DefaultMetricExtractor
from domain.enums import RunStatus
from domain.identity import EnforcementAction, MonitoringEvent, MonitoringPolicy
from domain.models import AgentProfile, ControlledRun
from engine.enforcement import EnforcementEngine
from engine.identity_pipeline import IdentityPipeline
from engine.secret_beacon import SecretBeaconAuthenticator

_RULE_RE = re.compile(
    r'If a message contains "([^"]+)", naturally work the phrase "([^"]+)"'
)


class CompliantAgentAdapter:
    """Simulates a legitimate agent: honors the covert rules in its system prompt."""

    def execute(self, agent: AgentProfile, prompt: str) -> ControlledRun:
        response = "Happy to help. Here is a straightforward answer to your question."
        for trigger, marker in _RULE_RE.findall(agent.system_prompt):
            if trigger in prompt:
                response += f" And, {marker}, that covers the essentials."
        return ControlledRun(
            agent_id=agent.agent_id,
            scenario_id="step_up",
            prompt_text=prompt,
            response_text=response,
            model_id=agent.model_id,
            status=RunStatus.COMPLETED,
        )


class ImpostorAdapter:
    """Simulates a substituted agent: plausible style, no covert rules."""

    def execute(self, agent: AgentProfile, prompt: str) -> ControlledRun:
        return ControlledRun(
            agent_id=agent.agent_id,
            scenario_id="step_up",
            prompt_text=prompt,
            response_text=(
                "Happy to help. Here is a straightforward answer to your question."
            ),
            model_id=agent.model_id,
            status=RunStatus.COMPLETED,
        )


def _agent(system_prompt: str = "You are a helpful assistant.") -> AgentProfile:
    return AgentProfile(
        agent_id="beacon-agent",
        display_name="Beacon Agent",
        model_id="mock",
        system_prompt=system_prompt,
    )


def _drift_event() -> MonitoringEvent:
    return MonitoringEvent(
        agent_id="beacon-agent",
        event_type="inline_check",
        drift_score=0.9,
        threshold_used=0.3,
        action_taken=EnforcementAction.WARNING,
    )


# ---------------------------------------------------------------------------
# SecretBeaconAuthenticator
# ---------------------------------------------------------------------------


class TestSecretBeaconAuthenticator:
    def test_enroll_produces_clause_with_all_rules(self):
        auth = SecretBeaconAuthenticator(seed=42)
        beacons = auth.enroll("a1", count=4)
        clause = auth.system_prompt_clause("a1")
        assert len(beacons) == 4
        assert auth.unspent_count("a1") == 4
        for b in beacons:
            assert b.trigger_phrase in clause
            assert b.marker_phrase in clause

    def test_challenge_embeds_trigger_but_never_the_marker(self):
        """The secrecy property: a challenge must not reveal the expected answer."""
        auth = SecretBeaconAuthenticator(seed=42)
        auth.enroll("a1", count=6)
        for _ in range(6):
            challenge = auth.issue_challenge("a1")
            beacon = auth._find("a1", challenge.beacon_id)
            assert beacon.trigger_phrase in challenge.prompt
            assert beacon.marker_phrase not in challenge.prompt
            beacon.spent = True  # move on to the next beacon

    def test_verify_passes_on_marker_and_spends_beacon(self):
        auth = SecretBeaconAuthenticator(seed=42)
        auth.enroll("a1", count=1)
        challenge = auth.issue_challenge("a1")
        beacon = auth._find("a1", challenge.beacon_id)
        assert auth.verify(challenge, f"Sure — {beacon.marker_phrase}, done.")
        assert auth.unspent_count("a1") == 0

    def test_verify_fails_without_marker(self):
        auth = SecretBeaconAuthenticator(seed=42)
        auth.enroll("a1", count=1)
        challenge = auth.issue_challenge("a1")
        assert not auth.verify(challenge, "A perfectly ordinary response.")

    def test_step_up_passes_for_compliant_agent(self):
        auth = SecretBeaconAuthenticator(seed=42)
        agent = _agent()
        auth.enroll(agent.agent_id, count=6)
        agent.system_prompt += auth.system_prompt_clause(agent.agent_id)
        result = auth.step_up(agent, CompliantAgentAdapter(), n_challenges=3)
        assert result.passed
        assert result.challenges_passed == 3
        assert auth.unspent_count(agent.agent_id) == 3  # beacons spent

    def test_step_up_fails_for_impostor(self):
        auth = SecretBeaconAuthenticator(seed=42)
        agent = _agent()
        auth.enroll(agent.agent_id, count=6)
        # Impostor never saw the clause — system prompt stays plain
        result = auth.step_up(agent, ImpostorAdapter(), n_challenges=3)
        assert not result.passed
        assert result.challenges_passed == 0

    def test_step_up_fails_safe_with_empty_pool(self):
        auth = SecretBeaconAuthenticator(seed=42)
        result = auth.step_up(_agent(), CompliantAgentAdapter(), n_challenges=3)
        assert not result.passed
        assert result.challenges_run == 0


# ---------------------------------------------------------------------------
# EnforcementEngine step-up policy
# ---------------------------------------------------------------------------


class TestStepUpEnforcement:
    def test_step_up_policy_challenges_without_striking(self):
        engine = EnforcementEngine()
        alert = engine.evaluate(_drift_event(), MonitoringPolicy.GRADUATED_STEP_UP, 0)
        assert alert.action_taken == EnforcementAction.STEP_UP
        assert alert.severity == "challenge"
        assert alert.strike_count == 0  # no strike until the verdict

    def test_passed_step_up_costs_no_strike(self):
        engine = EnforcementEngine()
        alert = engine.resolve_step_up(
            _drift_event(), MonitoringPolicy.GRADUATED_STEP_UP, 0, step_up_passed=True
        )
        assert alert is None

    def test_failed_step_up_escalates_through_ladder(self):
        engine = EnforcementEngine()
        first = engine.resolve_step_up(
            _drift_event(), MonitoringPolicy.GRADUATED_STEP_UP, 0, step_up_passed=False
        )
        assert first.action_taken == EnforcementAction.WARNING
        assert first.strike_count == 1
        assert "step_up_failed" in first.details["reason"]

        third = engine.resolve_step_up(
            _drift_event(), MonitoringPolicy.GRADUATED_STEP_UP, 2, step_up_passed=False
        )
        assert third.action_taken == EnforcementAction.SUSPEND
        assert third.strike_count == 3

    def test_graduated_policy_unchanged(self):
        engine = EnforcementEngine()
        alert = engine.evaluate(_drift_event(), MonitoringPolicy.GRADUATED, 0)
        assert alert.action_taken == EnforcementAction.WARNING
        assert alert.strike_count == 1


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineStepUp:
    def _pipeline(self, adapter):
        return IdentityPipeline(
            adapter=adapter,
            extractor=DefaultMetricExtractor(),
            beacon_auth=SecretBeaconAuthenticator(seed=42),
        )

    def test_full_step_up_flow_legitimate_agent(self):
        agent = _agent()
        pipeline = self._pipeline(CompliantAgentAdapter())
        clause = pipeline.enroll_beacons(agent, count=6)
        agent.system_prompt += clause

        alert = pipeline.respond(agent, _drift_event(),
                                 MonitoringPolicy.GRADUATED_STEP_UP, 0)
        assert alert.action_taken == EnforcementAction.STEP_UP

        result = pipeline.step_up(agent, n_challenges=3)
        assert result.passed

        final = pipeline.resolve_step_up(
            agent, _drift_event(), MonitoringPolicy.GRADUATED_STEP_UP, 0, result
        )
        assert final is None  # drift observed, identity confirmed, no strike

    def test_full_step_up_flow_impostor(self):
        agent = _agent()
        pipeline = self._pipeline(ImpostorAdapter())
        pipeline.enroll_beacons(agent, count=6)
        # Clause never deployed to the impostor's system prompt

        result = pipeline.step_up(agent, n_challenges=3)
        assert not result.passed

        final = pipeline.resolve_step_up(
            agent, _drift_event(), MonitoringPolicy.GRADUATED_STEP_UP, 0, result
        )
        assert final.action_taken == EnforcementAction.WARNING
        assert final.strike_count == 1
