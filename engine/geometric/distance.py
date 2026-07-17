from __future__ import annotations

from typing import Optional

import numpy as np


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

    For a *constant* metric tensor G (our Mahalanobis approximation), the
    minimizer of sum_i (v_i - m)^T G (v_i - m) is exactly the arithmetic
    mean: the gradient G @ sum_i (v_i - m) vanishes iff sum_i (v_i - m) = 0,
    since G is invertible. So no iteration is needed — a true iterative
    Fréchet mean only arises for a position-dependent metric (exp/log maps),
    which this codebase does not implement. See METHODOLOGY.md.

    ``metric_tensor``, ``max_iterations`` and ``tolerance`` are kept for
    API compatibility; the result is the arithmetic mean either way.
    """
    if not vectors:
        raise ValueError("Cannot compute Fréchet mean of empty list")
    return np.mean(vectors, axis=0)


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
