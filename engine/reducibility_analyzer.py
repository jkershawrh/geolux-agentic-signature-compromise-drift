from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from domain.enums import MetricDimension, Reducibility
from domain.metrics import MetricMeasurement
from domain.reducibility import ReducibilityClassification
from engine.geometric.embedding import metrics_to_vector


class ReducibilityAnalyzer:
    """Classify each metric as computationally reducible or irreducible.

    Reducible metrics are predictable across identical inputs — they form
    the stable core of a geometric signature. Irreducible metrics are
    inherently noisy and should be down-weighted or excluded.

    Based on Wolfram's Computational Reducibility theory: only reducible
    aspects of agent behavior can form reliable signatures.
    """

    def __init__(
        self,
        reducible_variance_threshold: float = 0.01,
        irreducible_variance_threshold: float = 0.1,
        min_samples: int = 5,
    ):
        self._reducible_threshold = reducible_variance_threshold
        self._irreducible_threshold = irreducible_variance_threshold
        self._min_samples = min_samples

    def analyze(
        self,
        metrics_per_run: list[list[MetricMeasurement]],
        agent_id: str,
    ) -> list[ReducibilityClassification]:
        """Analyze metrics across multiple runs to classify reducibility.

        All runs should use identical inputs (same prompts, same agent config)
        so that variance reflects inherent noise, not input variation.
        """
        if len(metrics_per_run) < self._min_samples:
            raise ValueError(
                f"Need at least {self._min_samples} runs, got {len(metrics_per_run)}"
            )

        vectors = np.array([metrics_to_vector(m) for m in metrics_per_run])

        from domain.metrics import METRIC_DEFINITIONS

        classifications = []
        col_idx = 0

        for dim in MetricDimension:
            for metric_name in METRIC_DEFINITIONS[dim]:
                column = vectors[:, col_idx]
                classification = self._classify_metric(
                    agent_id=agent_id,
                    dimension=dim,
                    metric_name=metric_name,
                    values=column,
                )
                classifications.append(classification)
                col_idx += 1

        return classifications

    def _classify_metric(
        self,
        agent_id: str,
        dimension: MetricDimension,
        metric_name: str,
        values: np.ndarray,
    ) -> ReducibilityClassification:
        """Classify a single metric based on its variance across runs."""
        variance = float(np.var(values))
        mean = float(np.mean(values))
        std = float(np.std(values))

        cv = std / max(abs(mean), 1e-10)

        autocorr = self._autocorrelation(values)

        if variance <= self._reducible_threshold:
            reducibility = Reducibility.REDUCIBLE
            predictability = 1.0 - min(variance / self._reducible_threshold, 1.0)
        elif variance >= self._irreducible_threshold:
            reducibility = Reducibility.IRREDUCIBLE
            predictability = max(0.0, 1.0 - variance)
        else:
            reducibility = Reducibility.CONDITIONALLY_REDUCIBLE
            ratio = (variance - self._reducible_threshold) / (
                self._irreducible_threshold - self._reducible_threshold
            )
            predictability = 1.0 - ratio

        predictability = max(0.0, min(1.0, predictability))

        evidence: dict[str, Any] = {
            "variance": variance,
            "mean": mean,
            "std": std,
            "coefficient_of_variation": cv,
            "autocorrelation_lag1": autocorr,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

        return ReducibilityClassification(
            classification_id=str(uuid.uuid4()),
            agent_id=agent_id,
            dimension=dimension,
            metric_name=metric_name,
            reducibility=reducibility,
            predictability_score=predictability,
            variance=variance,
            evidence=evidence,
            sample_size=len(values),
        )

    def _autocorrelation(self, values: np.ndarray) -> float:
        """Compute lag-1 autocorrelation. High = temporally predictable."""
        if len(values) < 3:
            return 0.0
        mean = np.mean(values)
        centered = values - mean
        var = np.var(values)
        if var == 0:
            return 1.0
        autocov = np.mean(centered[:-1] * centered[1:])
        return float(autocov / var)

    def get_reducible_mask(
        self, classifications: list[ReducibilityClassification]
    ) -> list[bool]:
        """Return a boolean mask indicating which metrics are reducible.

        Reducible and conditionally reducible metrics are included;
        irreducible metrics are excluded from signatures.
        """
        return [
            c.reducibility != Reducibility.IRREDUCIBLE
            for c in classifications
        ]

    def compute_fisher_ratios(
        self,
        agent_a_vectors: np.ndarray,  # shape (n_runs, n_metrics)
        agent_b_vectors: np.ndarray,
    ) -> dict[str, float]:
        """Compute Fisher's Linear Discriminant ratio per metric.

        High ratio = metric separates agents well.
        Low ratio = metric is noise for agent discrimination.
        """
        from domain.metrics import ALL_METRIC_NAMES

        ratios = {}
        for i, name in enumerate(ALL_METRIC_NAMES):
            mu_a = float(np.mean(agent_a_vectors[:, i]))
            mu_b = float(np.mean(agent_b_vectors[:, i]))
            var_a = float(np.var(agent_a_vectors[:, i], ddof=1)) if agent_a_vectors.shape[0] > 1 else 0.0
            var_b = float(np.var(agent_b_vectors[:, i], ddof=1)) if agent_b_vectors.shape[0] > 1 else 0.0
            ratios[name] = (mu_a - mu_b) ** 2 / (var_a + var_b + 1e-10)
        return ratios

    def get_discriminative_mask(
        self,
        fisher_ratios: dict[str, float],
        top_k: int = 12,
    ) -> list[bool]:
        """Return a mask keeping only the top-k metrics by Fisher ratio."""
        from domain.metrics import ALL_METRIC_NAMES

        sorted_metrics = sorted(fisher_ratios.items(), key=lambda x: -x[1])
        top_names = {name for name, _ in sorted_metrics[:top_k]}
        return [name in top_names for name in ALL_METRIC_NAMES]

    def summary(
        self, classifications: list[ReducibilityClassification]
    ) -> dict[str, int]:
        """Count metrics by reducibility classification."""
        counts = {r.value: 0 for r in Reducibility}
        for c in classifications:
            counts[c.reducibility.value] += 1
        return counts
