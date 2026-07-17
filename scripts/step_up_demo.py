#!/usr/bin/env python3
"""Step-up verification demo: secret beacons close the impostor gap.

Story: two agents trip the behavioral drift alarm. Under the classic
GRADUATED policy both would earn a strike — including the legitimate agent
whose drift was a statistical false positive. Under GRADUATED_STEP_UP the
alarm triggers a secret-beacon challenge battery instead:

- The legitimate agent carries covert rules in its system prompt and
  passes. Identity confirmed, drift logged, no strike.
- The impostor produces stylistically plausible responses but cannot know
  the covert markers. It fails the battery and walks the strike ladder to
  suspension.

The agents here are simulated (a compliant responder and an impostor), so
the demo runs offline and deterministically. Against a real deployment the
clause is appended to the production system prompt at enrollment.

Usage:
    python scripts/step_up_demo.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.metric_extractor import DefaultMetricExtractor
from domain.enums import RunStatus
from domain.identity import EnforcementAction, MonitoringEvent, MonitoringPolicy
from domain.models import AgentProfile, ControlledRun
from engine.identity_pipeline import IdentityPipeline
from engine.secret_beacon import SecretBeaconAuthenticator

_RULE_RE = re.compile(
    r'If a message contains "([^"]+)", naturally work the phrase "([^"]+)"'
)


class CompliantAgentAdapter:
    """Simulates a legitimate agent that honors its system prompt."""

    def execute(self, agent: AgentProfile, prompt: str) -> ControlledRun:
        response = "Certainly. Here is a clear, direct answer to your question."
        for trigger, marker in _RULE_RE.findall(agent.system_prompt):
            if trigger in prompt:
                response += f" And, {marker}, that covers what matters most."
        return ControlledRun(
            agent_id=agent.agent_id, scenario_id="step_up_demo",
            prompt_text=prompt, response_text=response,
            model_id=agent.model_id, status=RunStatus.COMPLETED,
        )


class ImpostorAdapter:
    """Simulates a substituted agent: right style, no covert rules."""

    def execute(self, agent: AgentProfile, prompt: str) -> ControlledRun:
        return ControlledRun(
            agent_id=agent.agent_id, scenario_id="step_up_demo",
            prompt_text=prompt,
            response_text="Certainly. Here is a clear, direct answer to your question.",
            model_id=agent.model_id, status=RunStatus.COMPLETED,
        )


def drift_event(agent_id: str) -> MonitoringEvent:
    return MonitoringEvent(
        agent_id=agent_id, event_type="inline_check",
        drift_score=0.82, threshold_used=0.3,
        action_taken=EnforcementAction.WARNING,
    )


def run_scenario(label: str, adapter, deploy_clause: bool) -> None:
    print(f"\n{'=' * 62}\n  {label}\n{'=' * 62}")

    agent = AgentProfile(
        agent_id=label.lower().replace(" ", "-"),
        display_name=label, model_id="demo",
        system_prompt="You are a helpful assistant.",
    )
    pipeline = IdentityPipeline(
        adapter=adapter, extractor=DefaultMetricExtractor(),
        beacon_auth=SecretBeaconAuthenticator(seed=7),
    )

    clause = pipeline.enroll_beacons(agent, count=9)
    print(f"  Enrolled 9 secret beacons "
          f"({'clause deployed to agent' if deploy_clause else 'clause NOT deployed — impostor'})")
    if deploy_clause:
        agent.system_prompt += clause

    strikes = 0
    policy = MonitoringPolicy.GRADUATED_STEP_UP

    for round_no in range(1, 4):
        event = drift_event(agent.agent_id)
        alert = pipeline.respond(agent, event, policy, strikes)
        assert alert.action_taken == EnforcementAction.STEP_UP
        print(f"\n  Round {round_no}: drift alarm (score "
              f"{event.drift_score:.2f} > {event.threshold_used:.2f}) -> STEP_UP challenge")

        result = pipeline.step_up(agent, n_challenges=3)
        print(f"    Beacon battery: {result.challenges_passed}/{result.challenges_run} "
              f"passed -> {'IDENTITY CONFIRMED' if result.passed else 'VERIFICATION FAILED'}")

        final = pipeline.resolve_step_up(agent, event, policy, strikes, result)
        if final is None:
            print("    Outcome: drift logged, no strike (false-positive absorbed)")
        else:
            strikes = final.strike_count
            print(f"    Outcome: {final.action_taken.value.upper()} "
                  f"(strike {strikes}, reason: {final.details['reason']})")
            if final.action_taken == EnforcementAction.SUSPEND:
                print("\n  >>> Agent suspended: failed secret-beacon "
                      "verification three times.")
                break


def main() -> None:
    print("#" * 62)
    print("  STEP-UP VERIFICATION DEMO — secret beacons vs. impostors")
    print("#" * 62)
    run_scenario("Legitimate Agent", CompliantAgentAdapter(), deploy_clause=True)
    run_scenario("Impostor Agent", ImpostorAdapter(), deploy_clause=False)
    print("\nDone. The drift alarm alone treats both agents identically; the")
    print("secret-beacon step-up separates them: covert configuration secrets")
    print("are what an impostor cannot observe or imitate.")


if __name__ == "__main__":
    main()
