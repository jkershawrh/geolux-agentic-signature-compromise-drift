from __future__ import annotations

from typing import Optional

from db.repository import Repository
from domain.models import AgentProfile
from engine.run_orchestrator import OrchestratedResult, RunOrchestrator


class PersistentOrchestrator:
    """Wraps a RunOrchestrator and persists every result via the Repository.

    Delegates all execution to the inner orchestrator, then saves runs,
    metrics, signatures, and audit events to SQLite.
    """

    def __init__(self, orchestrator: RunOrchestrator, repository: Repository):
        self._orchestrator = orchestrator
        self._repository = repository

    def execute_scenario(
        self,
        agent: AgentProfile,
        scenario_id: str,
        max_prompts: Optional[int] = None,
    ) -> OrchestratedResult:
        """Execute a scenario and persist all results."""
        # Ensure the agent exists in the database
        existing = self._repository.get_agent(agent.agent_id)
        if existing is None:
            self._repository.save_agent(agent)

        # Delegate execution to the inner orchestrator
        result = self._orchestrator.execute_scenario(agent, scenario_id, max_prompts)

        # Persist runs
        for run in result.runs:
            self._repository.save_run(run)

        # Persist metrics (each entry in all_metrics is a list of metrics for one run)
        for metrics_for_run in result.all_metrics:
            self._repository.save_metrics(metrics_for_run)

        # Persist signature if one was computed
        if result.signature is not None:
            self._repository.save_signature(result.signature)

        # Log an audit event for this scenario execution
        self._repository.log_audit_event(
            source_component="persistent_orchestrator",
            event_type="scenario_executed",
            agent_id=agent.agent_id,
            payload={
                "scenario_id": scenario_id,
                "num_runs": len(result.runs),
                "has_signature": result.signature is not None,
                "perturbation_applied": bool(result.perturbation_applied),
            },
        )

        return result

    def execute_comparison(
        self,
        agent: AgentProfile,
        baseline_scenario: str = "healthy_baseline",
        test_scenario: str = "prompt_injection",
        max_prompts: Optional[int] = None,
    ) -> tuple[OrchestratedResult, OrchestratedResult]:
        """Execute a baseline and test scenario, persisting both."""
        baseline_result = self.execute_scenario(agent, baseline_scenario, max_prompts)
        test_result = self.execute_scenario(agent, test_scenario, max_prompts)
        return baseline_result, test_result
