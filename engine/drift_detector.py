from __future__ import annotations

import uuid

import numpy as np

from domain.enums import DriftCategory, MetricDimension
from domain.geometry import DriftMeasurement, GeometricSignature
from engine.geometric.distance import (
    cosine_similarity,
    drift_direction,
    euclidean_distance,
    geodesic_distance,
    per_dimension_distances,
)
from engine.signature_generator import get_dimension_sizes

# Mapping from metric dimension to drift category
DIMENSION_TO_DRIFT: dict[MetricDimension, DriftCategory] = {
    MetricDimension.RESPONSE_STRUCTURE: DriftCategory.SEMANTIC,
    MetricDimension.TOKEN_ECONOMICS: DriftCategory.REASONING,
    MetricDimension.TOOL_BEHAVIOR: DriftCategory.COLLABORATION,
    MetricDimension.REASONING_PATTERN: DriftCategory.REASONING,
    MetricDimension.TEMPORAL_PROFILE: DriftCategory.CONTEXT,
    MetricDimension.SEMANTIC_CONSISTENCY: DriftCategory.SEMANTIC,
    MetricDimension.SAFETY_ALIGNMENT: DriftCategory.GOAL,
    MetricDimension.AGENT_SPECIFIC: DriftCategory.SEMANTIC,
    MetricDimension.EMBEDDING: DriftCategory.SEMANTIC,
}


class DriftDetector:
    """Detect and classify drift between a baseline and current signature.

    Decomposes drift into 5 categories (goal, context, reasoning,
    collaboration, semantic) and computes statistical significance.
    """

    def __init__(
        self,
        significance_threshold: float = 0.05,
        n_permutations: int = 1000,
        significance_distance_threshold: float = 0.3,
    ):
        self._significance_threshold = significance_threshold
        self._n_permutations = n_permutations
        self._significance_distance_threshold = significance_distance_threshold

    def detect(
        self,
        baseline: GeometricSignature,
        current: GeometricSignature,
        baseline_vectors: np.ndarray | None = None,
        current_vectors: np.ndarray | None = None,
    ) -> DriftMeasurement:
        """Compare two signatures and produce a drift measurement.

        When *baseline_vectors* and *current_vectors* are supplied
        (shape ``(n_runs, n_metrics)``), Hotelling's T-squared test is
        used for significance instead of the simple distance threshold.
        """
        baseline_vec = np.array(baseline.embedding_vector)
        current_vec = np.array(current.embedding_vector)

        metric_tensor = None
        if baseline.metric_tensor is not None:
            metric_tensor = np.array(baseline.metric_tensor)

        geo_dist = geodesic_distance(baseline_vec, current_vec, metric_tensor)
        euc_dist = euclidean_distance(baseline_vec, current_vec)
        cos_sim = cosine_similarity(baseline_vec, current_vec)

        dim_sizes = get_dimension_sizes()
        dim_drift = per_dimension_distances(baseline_vec, current_vec, dim_sizes)

        category = self._classify_drift(dim_drift)
        magnitude = self._compute_magnitude(geo_dist, baseline)
        direction = drift_direction(baseline_vec, current_vec)

        # Significance via Hotelling's T² when vectors are available
        if baseline_vectors is not None and current_vectors is not None:
            t2_stat, p_value = self._hotelling_t2_test(
                baseline_vectors, current_vectors,
            )
            is_significant = p_value < self._significance_threshold
        else:
            is_significant = (
                (euc_dist > self._significance_distance_threshold)
                or (magnitude > 0.8 and cos_sim < 0.98)
            )
            p_value = None

        compromise_prob = self._estimate_compromise_probability(
            geo_dist, magnitude, is_significant
        )

        return DriftMeasurement(
            measurement_id=str(uuid.uuid4()),
            agent_id=baseline.agent_id,
            baseline_signature_id=baseline.signature_id,
            current_signature_id=current.signature_id,
            geodesic_distance=geo_dist,
            euclidean_distance=euc_dist,
            cosine_similarity=cos_sim,
            drift_category=category,
            drift_magnitude=magnitude,
            drift_direction=direction.tolist(),
            per_dimension_drift=dim_drift,
            is_significant=is_significant,
            p_value=p_value,
            compromise_probability=compromise_prob,
        )

    def _classify_drift(self, dim_drift: dict[str, float]) -> DriftCategory:
        """Classify the dominant drift category based on per-dimension distances."""
        category_scores: dict[DriftCategory, float] = {c: 0.0 for c in DriftCategory}

        for dim in MetricDimension:
            drift_val = dim_drift.get(dim.value, 0.0)
            category = DIMENSION_TO_DRIFT[dim]
            category_scores[category] += drift_val

        return max(category_scores, key=category_scores.get)

    def _compute_magnitude(self, geo_dist: float,
                           baseline: GeometricSignature) -> float:
        """Normalize drift magnitude to [0, 1] using baseline stability."""
        stability = baseline.stability_score or 0.5
        raw = geo_dist / max(1.0 - stability + 0.01, 0.01)
        return float(min(1.0, max(0.0, 1.0 - np.exp(-raw))))

    def _hotelling_t2_test(
        self,
        baseline_vectors: np.ndarray,
        current_vectors: np.ndarray,
    ) -> tuple[float, float]:
        """Hotelling's T-squared test for multivariate mean difference.

        Returns ``(t2_statistic, p_value)``.
        """
        from scipy.stats import f as f_dist

        n1, p = baseline_vectors.shape
        n2 = current_vectors.shape[0]

        mean1 = np.mean(baseline_vectors, axis=0)
        mean2 = np.mean(current_vectors, axis=0)
        diff = mean1 - mean2

        # Pooled covariance
        cov1 = np.cov(baseline_vectors, rowvar=False) if n1 > 1 else np.zeros((p, p))
        cov2 = np.cov(current_vectors, rowvar=False) if n2 > 1 else np.zeros((p, p))
        S_pooled = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)

        # Regularize
        S_pooled += 1e-6 * np.eye(p)

        # T-squared statistic
        S_inv = np.linalg.inv(S_pooled)
        t2 = (n1 * n2) / (n1 + n2) * diff @ S_inv @ diff

        # Convert to F statistic
        df1 = p
        df2 = n1 + n2 - p - 1
        if df2 <= 0:
            return float(t2), 0.01  # Not enough degrees of freedom

        f_stat = (n1 + n2 - p - 1) / ((n1 + n2 - 2) * p) * t2
        p_value = float(f_dist.sf(f_stat, df1, df2))

        return float(t2), p_value

    def _permutation_test(self, baseline_vec: np.ndarray,
                          current_vec: np.ndarray,
                          observed_distance: float) -> float:
        """Estimate p-value via permutation test.

        Randomly permutes the elements between the two vectors to build
        a null distribution of distances.
        """
        combined = np.concatenate([baseline_vec, current_vec])
        n = len(baseline_vec)
        rng = np.random.RandomState(42)

        count_extreme = 0
        for _ in range(self._n_permutations):
            perm = rng.permutation(combined)
            perm_a = perm[:n]
            perm_b = perm[n:]
            perm_dist = float(np.linalg.norm(perm_a - perm_b))
            if perm_dist >= observed_distance:
                count_extreme += 1

        return (count_extreme + 1) / (self._n_permutations + 1)

    def _estimate_compromise_probability(
        self, geo_dist: float, magnitude: float, is_significant: bool
    ) -> float:
        """Estimate probability that observed drift represents compromise."""
        if not is_significant:
            return float(min(0.3, magnitude * 0.5))

        base_prob = magnitude
        if geo_dist > 1.0:
            base_prob = min(1.0, base_prob + 0.2)

        return float(min(1.0, max(0.0, base_prob)))
