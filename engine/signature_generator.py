from __future__ import annotations

import uuid
from typing import Optional

import numpy as np

from domain.enums import MetricDimension, SignatureType
from domain.geometry import GeometricSignature
from domain.metrics import METRIC_DEFINITIONS, MetricMeasurement
from engine.geometric.distance import frechet_mean
from engine.geometric.embedding import MetricVectorBuilder, metrics_to_vector
from engine.geometric.manifold import reduce_to_manifold
from engine.geometric.riemannian import compute_metric_tensor


def get_dimension_sizes() -> list[int]:
    """Get the number of metrics per dimension, in MetricDimension enum order."""
    return [len(METRIC_DEFINITIONS[dim]) for dim in MetricDimension]


class SignatureGenerator:
    """Generates geometric signatures from collections of metric measurements.

    A signature captures the geometric "fingerprint" of an agent's behavior
    by computing the centroid and shape of its metric vectors in a
    Riemannian manifold space.
    """

    def __init__(self, min_runs: int = 5, manifold_method: str = "pca"):
        self._min_runs = min_runs
        self._manifold_method = manifold_method

    def generate(
        self,
        agent_id: str,
        metrics_per_run: list[list[MetricMeasurement]],
        run_ids: list[str],
        signature_type: SignatureType = SignatureType.SNAPSHOT,
        reducibility_mask: list[bool] | None = None,
    ) -> GeometricSignature:
        """Generate a geometric signature from multiple runs' metrics.

        Each element of metrics_per_run is the full set of 36 metrics
        extracted from one controlled run.

        If *reducibility_mask* is provided it must have one entry per metric
        (length 36).  Metrics marked ``False`` (irreducible / noisy) are
        zeroed out before computing the centroid, covariance, and all
        downstream geometry.  This filters noise from the signature.
        """
        if len(metrics_per_run) < self._min_runs:
            raise ValueError(f"Need at least {self._min_runs} runs, got {len(metrics_per_run)}")

        builder = MetricVectorBuilder()
        for run_metrics in metrics_per_run:
            builder.add_metrics(run_metrics)

        centroid = builder.get_centroid()
        covariance = builder.get_covariance()
        all_vectors = builder.get_all_vectors()

        # Apply reducibility mask — zero out irreducible / noisy dimensions.
        if reducibility_mask is not None:
            mask_array = np.array(
                [1.0 if m else 0.0 for m in reducibility_mask],
                dtype=np.float64,
            )
            all_vectors = all_vectors * mask_array  # broadcast per-row
            centroid = centroid * mask_array
            covariance = covariance * np.outer(mask_array, mask_array)

        metric_tensor = compute_metric_tensor(covariance)

        riemannian_centroid = frechet_mean(
            [v for v in all_vectors], metric_tensor=metric_tensor
        )

        if all_vectors.shape[0] >= 3:
            n_components = min(2, all_vectors.shape[1])
            manifold_coords = reduce_to_manifold(
                all_vectors, n_components=n_components,
                method=self._manifold_method
            )
            centroid_manifold = manifold_coords.mean(axis=0)
        else:
            centroid_manifold = riemannian_centroid[:2]

        stability = self._compute_stability(all_vectors, riemannian_centroid, metric_tensor)

        metric_snapshot = {}
        for run_metrics in metrics_per_run:
            for m in run_metrics:
                if m.metric_name not in metric_snapshot:
                    metric_snapshot[m.metric_name] = []
                metric_snapshot[m.metric_name].append(m.normalized_value)

        metric_means = {k: float(np.mean(v)) for k, v in metric_snapshot.items()}

        # Compute cross-run variance for metrics that are undefined per-run
        if len(metrics_per_run) >= 2:
            response_lengths = [m.normalized_value for run_metrics in metrics_per_run
                                for m in run_metrics if m.metric_name == "avg_response_length"]
            latencies = [m.normalized_value for run_metrics in metrics_per_run
                         for m in run_metrics if m.metric_name == "mean_latency_ms"]
            if response_lengths:
                metric_means["response_length_variance"] = float(np.var(response_lengths))
            if latencies:
                metric_means["latency_variance"] = float(np.var(latencies))

        return GeometricSignature(
            signature_id=str(uuid.uuid4()),
            agent_id=agent_id,
            signature_type=signature_type,
            embedding_vector=riemannian_centroid.tolist(),
            embedding_dimension=len(riemannian_centroid),
            manifold_coordinates=centroid_manifold.tolist(),
            metric_tensor=metric_tensor.tolist(),
            metric_snapshot=metric_means,
            run_ids=run_ids,
            num_runs=len(run_ids),
            computation_method=self._manifold_method,
            stability_score=stability,
        )

    def _compute_stability(self, vectors: np.ndarray, centroid: np.ndarray,
                           metric_tensor: np.ndarray) -> float:
        """Compute stability score [0,1] measuring how tight the signature is.

        Higher stability = more consistent behavior = more reliable signature.
        Uses the average Mahalanobis distance from centroid, normalized to [0,1].
        """
        distances = []
        for v in vectors:
            diff = v - centroid
            dist = float(np.sqrt(np.abs(diff @ metric_tensor @ diff)))
            distances.append(dist)

        avg_distance = np.mean(distances)
        stability = float(np.exp(-avg_distance))
        return max(0.0, min(1.0, stability))
