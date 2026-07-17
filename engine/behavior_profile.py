"""Behavioral profile reporting: turn stored metrics into an interpretable
summary of how an agent behaves.

The metric extractor and signature generator answer machine questions
(vectors, distances). This module answers the human question — "what does
this agent's behavior look like, and how consistent is it?" — from the same
stored measurements: per-dimension averages, the most stable and most
volatile metrics, and an overall consistency score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from domain.metrics import MetricMeasurement


@dataclass
class MetricStat:
    metric_name: str
    dimension: str
    mean: float
    std: float
    n_samples: int


@dataclass
class BehaviorProfile:
    """Interpretable behavioral summary for one agent."""

    agent_id: str
    n_runs: int
    n_metrics: int
    # dimension -> {"mean": avg normalized value, "std": avg within-metric std}
    per_dimension: dict[str, dict[str, float]] = field(default_factory=dict)
    most_stable: list[MetricStat] = field(default_factory=list)
    most_variable: list[MetricStat] = field(default_factory=list)
    # 1.0 = perfectly repeatable behavior, 0.0 = noise
    consistency_score: float = 0.0


class BehaviorProfiler:
    """Build a BehaviorProfile from stored metric measurements."""

    def __init__(self, top_k: int = 5):
        self._top_k = top_k

    def profile(self, agent_id: str, metrics: list[MetricMeasurement]) -> BehaviorProfile:
        if not metrics:
            return BehaviorProfile(agent_id=agent_id, n_runs=0, n_metrics=0)

        # Group normalized values per metric across runs
        by_metric: dict[str, list[float]] = {}
        dim_of: dict[str, str] = {}
        run_ids = set()
        for m in metrics:
            by_metric.setdefault(m.metric_name, []).append(m.normalized_value)
            dim_of[m.metric_name] = (
                m.dimension.value if hasattr(m.dimension, "value") else str(m.dimension)
            )
            run_ids.add(m.run_id)

        stats = [
            MetricStat(
                metric_name=name,
                dimension=dim_of[name],
                mean=_mean(values),
                std=_std(values),
                n_samples=len(values),
            )
            for name, values in sorted(by_metric.items())
        ]

        # Per-dimension aggregates
        per_dimension: dict[str, dict[str, float]] = {}
        for dim in sorted({s.dimension for s in stats}):
            dim_stats = [s for s in stats if s.dimension == dim]
            per_dimension[dim] = {
                "mean": _mean([s.mean for s in dim_stats]),
                "std": _mean([s.std for s in dim_stats]),
                "metric_count": float(len(dim_stats)),
            }

        # Stability ranking only makes sense with repeated observations
        repeated = [s for s in stats if s.n_samples >= 2]
        by_std = sorted(repeated, key=lambda s: s.std)
        most_stable = by_std[: self._top_k]
        most_variable = list(reversed(by_std[-self._top_k:])) if by_std else []

        # Consistency: average within-metric std of normalized values,
        # mapped to [0, 1] where 0 std -> 1.0
        avg_std = _mean([s.std for s in repeated]) if repeated else 0.0
        consistency = math.exp(-4.0 * avg_std)

        return BehaviorProfile(
            agent_id=agent_id,
            n_runs=len(run_ids),
            n_metrics=len(stats),
            per_dimension=per_dimension,
            most_stable=most_stable,
            most_variable=most_variable,
            consistency_score=round(consistency, 4),
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))
