from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns value in [-1, 1].

    Result clamped to [-1, 1] to handle floating-point precision.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(max(-1.0, min(1.0, np.dot(a, b) / (norm_a * norm_b))))


def geodesic_distance(a: np.ndarray, b: np.ndarray,
                      metric_tensor: Optional[np.ndarray] = None) -> float:
    """Compute geodesic distance between two points.

    If a Riemannian metric tensor G is provided, computes the Mahalanobis-like
    distance: sqrt((a-b)^T G (a-b)). This approximates the geodesic distance
    on a Riemannian manifold when points are close.

    Without a metric tensor, falls back to Euclidean distance.

    Note: This computes Mahalanobis distance (sqrt of quadratic form with
    precision matrix), not true geodesic distance on a curved manifold.
    See METHODOLOGY.md.
    """
    diff = a - b
    if metric_tensor is not None:
        return float(np.sqrt(np.abs(diff @ metric_tensor @ diff)))
    return float(np.linalg.norm(diff))


def frechet_mean(vectors: list[np.ndarray],
                 metric_tensor: Optional[np.ndarray] = None,
                 max_iterations: int = 100,
                 tolerance: float = 1e-8) -> np.ndarray:
    """Compute the Fréchet mean (geometric center of mass).

    For Euclidean metric, this is simply the arithmetic mean.
    For a Riemannian metric tensor, uses iterative weighted averaging
    (gradient descent on the sum of squared geodesic distances).

    Note: With a metric tensor, this uses gradient descent on a quadratic
    form, not exp/log maps on a Riemannian manifold. Converges to the
    arithmetic mean for constant metric. See METHODOLOGY.md.
    """
    if not vectors:
        raise ValueError("Cannot compute Fréchet mean of empty list")

    if metric_tensor is None:
        return np.mean(vectors, axis=0)

    # Iterative Fréchet mean for Riemannian metric
    mean = np.mean(vectors, axis=0)
    for _ in range(max_iterations):
        gradient = np.zeros_like(mean)
        for v in vectors:
            diff = v - mean
            gradient += metric_tensor @ diff
        gradient /= len(vectors)

        new_mean = mean + gradient
        if np.linalg.norm(new_mean - mean) < tolerance:
            break
        mean = new_mean

    return mean


def per_dimension_distances(a: np.ndarray, b: np.ndarray,
                            dimension_sizes: list[int]) -> dict[str, float]:
    """Compute distance decomposed by metric dimension.

    dimension_sizes gives the number of metrics per dimension, in order
    matching MetricDimension enum.
    """
    from domain.enums import MetricDimension

    dims = list(MetricDimension)
    if len(dims) != len(dimension_sizes):
        raise ValueError("dimension_sizes must match number of MetricDimensions")

    result = {}
    offset = 0
    for dim, size in zip(dims, dimension_sizes):
        slice_a = a[offset:offset + size]
        slice_b = b[offset:offset + size]
        result[dim.value] = float(np.linalg.norm(slice_a - slice_b))
        offset += size

    return result


def drift_direction(baseline: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Compute the unit direction vector of drift from baseline to current."""
    diff = current - baseline
    norm = np.linalg.norm(diff)
    if norm == 0:
        return np.zeros_like(diff)
    return diff / norm
