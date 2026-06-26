from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from adapters.interfaces import InferenceAdapter, MetricExtractor
from domain.enums import AgentStatus, SignatureType
from domain.geometry import GeometricSignature
from domain.models import AgentProfile
from engine.baseline_engine import BaselineEngine, BaselineResult
from engine.geometric.distance import geodesic_distance
from engine.signature_generator import SignatureGenerator


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    agent_id: str
    success: bool
    old_baseline: GeometricSignature
    new_baseline: Optional[GeometricSignature]
    distance_from_old: float
    convergence_achieved: bool
    details: str


class RecoveryEngine:
    """Recovery procedures after an agent is flagged as compromised.

    Recovery involves re-establishing a clean baseline and validating
    that the agent is operating within expected parameters.
    """

    def __init__(
        self,
        adapter: InferenceAdapter,
        extractor: MetricExtractor,
        generator: Optional[SignatureGenerator] = None,
        recovery_distance_threshold: float = 0.3,
    ):
        self._adapter = adapter
        self._extractor = extractor
        self._generator = generator or SignatureGenerator(manifold_method="pca")
        self._recovery_threshold = recovery_distance_threshold

    def recover(
        self,
        agent: AgentProfile,
        old_baseline: GeometricSignature,
        prompts: list[str],
    ) -> RecoveryResult:
        """Attempt to recover an agent by re-establishing a baseline.

        Runs the agent on clean prompts, computes a new baseline,
        and validates it against the old baseline to ensure the agent
        has returned to normal operation.
        """
        baseline_engine = BaselineEngine(
            adapter=self._adapter,
            extractor=self._extractor,
            generator=self._generator,
            min_runs=len(prompts),
        )

        result = baseline_engine.establish_baseline(agent, prompts)

        old_vec = np.array(old_baseline.embedding_vector)
        new_vec = np.array(result.signature.embedding_vector)

        metric_tensor = None
        if old_baseline.metric_tensor is not None:
            metric_tensor = np.array(old_baseline.metric_tensor)

        distance = geodesic_distance(old_vec, new_vec, metric_tensor)
        is_close = distance <= self._recovery_threshold

        success = is_close and result.is_converged

        if success:
            details = (
                f"Recovery successful: new baseline is {distance:.4f} "
                f"geodesic distance from old (threshold={self._recovery_threshold}), "
                f"convergence={'achieved' if result.is_converged else 'not achieved'}"
            )
        else:
            reasons = []
            if not is_close:
                reasons.append(
                    f"distance={distance:.4f} exceeds threshold={self._recovery_threshold}"
                )
            if not result.is_converged:
                reasons.append("baseline did not converge")
            details = f"Recovery incomplete: {'; '.join(reasons)}"

        return RecoveryResult(
            agent_id=agent.agent_id,
            success=success,
            old_baseline=old_baseline,
            new_baseline=result.signature,
            distance_from_old=distance,
            convergence_achieved=result.is_converged,
            details=details,
        )
