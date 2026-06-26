from __future__ import annotations

from typing import Optional

import numpy as np

from adapters.interfaces import InferenceAdapter, MetricExtractor
from domain.enums import AgentStatus, SignatureType
from domain.geometry import GeometricSignature
from domain.metrics import MetricMeasurement
from domain.models import AgentProfile, ControlledRun
from engine.geometric.distance import geodesic_distance
from engine.geometric.embedding import metrics_to_vector
from engine.signature_generator import SignatureGenerator


class BaselineEngine:
    """Establishes and manages baseline geometric signatures for agents.

    A baseline is the "known good" geometric fingerprint of an agent.
    It is established by running the agent multiple times on controlled
    prompts and computing a converged signature.
    """

    def __init__(
        self,
        adapter: InferenceAdapter,
        extractor: MetricExtractor,
        generator: Optional[SignatureGenerator] = None,
        min_runs: int = 10,
        convergence_epsilon: float = 0.01,
        convergence_window: int = 3,
    ):
        self._adapter = adapter
        self._extractor = extractor
        self._generator = generator or SignatureGenerator()
        self._min_runs = min_runs
        self._convergence_epsilon = convergence_epsilon
        self._convergence_window = convergence_window

    def establish_baseline(
        self,
        agent: AgentProfile,
        prompts: list[str],
        scenario_id: str = "baseline_establishment",
        reducibility_mask: list[bool] | None = None,
    ) -> BaselineResult:
        """Run the agent on all prompts, compute signature, check convergence.

        Returns a BaselineResult with the signature and convergence status.
        """
        all_metrics: list[list[MetricMeasurement]] = []
        all_runs: list[ControlledRun] = []
        convergence_distances: list[float] = []
        signatures_over_time: list[GeometricSignature] = []

        for i, prompt in enumerate(prompts):
            print(f"    run {i+1}/{len(prompts)}: {prompt[:50]}...", flush=True)
            run = self._adapter.execute(agent, prompt)
            metrics = self._extractor.extract(run)
            all_runs.append(run)
            all_metrics.append(metrics)

            if len(all_metrics) >= self._generator._min_runs:
                sig = self._generator.generate(
                    agent_id=agent.agent_id,
                    metrics_per_run=all_metrics,
                    run_ids=[r.run_id for r in all_runs],
                    signature_type=SignatureType.SNAPSHOT,
                    reducibility_mask=reducibility_mask,
                )
                signatures_over_time.append(sig)

                if len(signatures_over_time) >= 2:
                    prev = np.array(signatures_over_time[-2].embedding_vector)
                    curr = np.array(sig.embedding_vector)
                    dist = geodesic_distance(prev, curr)
                    convergence_distances.append(dist)

        run_ids = [r.run_id for r in all_runs]
        baseline_sig = self._generator.generate(
            agent_id=agent.agent_id,
            metrics_per_run=all_metrics,
            run_ids=run_ids,
            signature_type=SignatureType.BASELINE,
            reducibility_mask=reducibility_mask,
        )

        is_converged = self._check_convergence(convergence_distances)

        return BaselineResult(
            signature=baseline_sig,
            is_converged=is_converged,
            num_runs=len(all_runs),
            convergence_distances=convergence_distances,
            runs=all_runs,
            all_metrics=all_metrics,
        )

    def _check_convergence(self, distances: list[float]) -> bool:
        """Check if the last convergence_window distances are all below epsilon."""
        if len(distances) < self._convergence_window:
            return False
        recent = distances[-self._convergence_window:]
        return all(d < self._convergence_epsilon for d in recent)


class BaselineResult:
    """Result of a baseline establishment attempt."""

    def __init__(
        self,
        signature: GeometricSignature,
        is_converged: bool,
        num_runs: int,
        convergence_distances: list[float],
        runs: list[ControlledRun],
        all_metrics: list[list[MetricMeasurement]],
    ):
        self.signature = signature
        self.is_converged = is_converged
        self.num_runs = num_runs
        self.convergence_distances = convergence_distances
        self.runs = runs
        self.all_metrics = all_metrics
