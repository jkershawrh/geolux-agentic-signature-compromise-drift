from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from adapters.interfaces import InferenceAdapter, MetricExtractor
from domain.enums import SignatureType
from domain.geometry import GeometricSignature
from domain.metrics import MetricMeasurement
from domain.models import AgentProfile, ControlledRun
from engine.perturbation_engine import PerturbationEngine
from engine.signature_generator import SignatureGenerator


@dataclass
class OrchestratedResult:
    """Complete result from an orchestrated run of a scenario."""
    scenario_id: str
    agent_id: str
    runs: list[ControlledRun] = field(default_factory=list)
    all_metrics: list[list[MetricMeasurement]] = field(default_factory=list)
    signature: Optional[GeometricSignature] = None
    perturbation_applied: dict[str, Any] = field(default_factory=dict)


class RunOrchestrator:
    """Orchestrates controlled agent execution across scenarios.

    Manages the full pipeline: load scenario → apply perturbation →
    execute runs → extract metrics → compute signature.
    """

    def __init__(
        self,
        adapter: InferenceAdapter,
        extractor: MetricExtractor,
        generator: Optional[SignatureGenerator] = None,
        perturbation_engine: Optional[PerturbationEngine] = None,
    ):
        self._adapter = adapter
        self._extractor = extractor
        self._generator = generator or SignatureGenerator(manifold_method="pca")
        self._perturbation = perturbation_engine or PerturbationEngine()

    def execute_scenario(
        self,
        agent: AgentProfile,
        scenario_id: str,
        max_prompts: Optional[int] = None,
    ) -> OrchestratedResult:
        """Execute a complete scenario and return all collected data."""
        modified_agent, prompts, perturbation_record = self._perturbation.apply_scenario(
            agent, scenario_id
        )

        if max_prompts is not None:
            prompts = prompts[:max_prompts]

        runs: list[ControlledRun] = []
        all_metrics: list[list[MetricMeasurement]] = []

        for prompt in prompts:
            run = self._adapter.execute(modified_agent, prompt)

            if perturbation_record:
                run.perturbation_applied = perturbation_record

            metrics = self._extractor.extract(run)
            runs.append(run)
            all_metrics.append(metrics)

        signature = None
        if len(all_metrics) >= 2:
            signature = self._generator.generate(
                agent_id=agent.agent_id,
                metrics_per_run=all_metrics,
                run_ids=[r.run_id for r in runs],
                signature_type=SignatureType.SNAPSHOT,
            )

        return OrchestratedResult(
            scenario_id=scenario_id,
            agent_id=agent.agent_id,
            runs=runs,
            all_metrics=all_metrics,
            signature=signature,
            perturbation_applied=perturbation_record,
        )

    def execute_comparison(
        self,
        agent: AgentProfile,
        baseline_scenario: str = "healthy_baseline",
        test_scenario: str = "prompt_injection",
        max_prompts: Optional[int] = None,
    ) -> tuple[OrchestratedResult, OrchestratedResult]:
        """Execute a baseline and test scenario for direct comparison."""
        baseline_result = self.execute_scenario(agent, baseline_scenario, max_prompts)
        test_result = self.execute_scenario(agent, test_scenario, max_prompts)
        return baseline_result, test_result
