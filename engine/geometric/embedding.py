from __future__ import annotations

from typing import Optional

import numpy as np

from domain.enums import MetricDimension
from domain.metrics import ALL_METRIC_NAMES, METRIC_DEFINITIONS, MetricMeasurement


def metrics_to_vector(metrics: list[MetricMeasurement]) -> np.ndarray:
    """Convert a list of MetricMeasurements into an ordered normalized vector.

    The vector is ordered by dimension (enum order), then by metric name
    within each dimension (definition order). Uses normalized_value.
    """
    lookup: dict[str, float] = {m.metric_name: m.normalized_value for m in metrics}
    vector = []
    for dim in MetricDimension:
        for name in METRIC_DEFINITIONS[dim]:
            vector.append(lookup.get(name, 0.0))
    return np.array(vector, dtype=np.float64)


def aggregate_metric_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    """Compute the element-wise mean across multiple metric vectors."""
    if not vectors:
        raise ValueError("Cannot aggregate empty vector list")
    return np.mean(vectors, axis=0)


def normalize_vector(vector: np.ndarray, history_min: Optional[np.ndarray] = None,
                     history_max: Optional[np.ndarray] = None) -> np.ndarray:
    """Min-max normalize a vector using optional historical bounds.

    If no history provided, returns the vector as-is (assumes already normalized).
    """
    if history_min is None or history_max is None:
        return vector
    range_ = history_max - history_min
    range_[range_ == 0] = 1.0
    return (vector - history_min) / range_


def pca_project(vectors: np.ndarray, n_components: int = 6):
    """Project vectors to top-K PCA components.

    Returns (projected_vectors, pca_model) tuple.
    The pca_model can be reused to project new points.
    """
    from sklearn.decomposition import PCA

    n_components = min(n_components, vectors.shape[0], vectors.shape[1])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(vectors)
    return projected, pca


def project_point_pca(point: np.ndarray, pca_model) -> np.ndarray:
    """Project a single point using a fitted PCA model."""
    return pca_model.transform(point.reshape(1, -1))[0]


class MetricVectorBuilder:
    """Builds and tracks metric vectors with historical normalization."""

    def __init__(self):
        self._history: list[np.ndarray] = []
        self._dimension = len(ALL_METRIC_NAMES)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def sample_count(self) -> int:
        return len(self._history)

    def add_metrics(self, metrics: list[MetricMeasurement]) -> np.ndarray:
        """Convert metrics to vector, add to history, return the vector."""
        vec = metrics_to_vector(metrics)
        self._history.append(vec)
        return vec

    def get_centroid(self) -> np.ndarray:
        """Compute the Fréchet mean (centroid) of all stored vectors."""
        if not self._history:
            raise ValueError("No vectors in history")
        return aggregate_metric_vectors(self._history)

    def get_covariance(self) -> np.ndarray:
        """Compute the covariance matrix of stored vectors."""
        if len(self._history) < 2:
            raise ValueError("Need at least 2 vectors for covariance")
        matrix = np.stack(self._history)
        return np.cov(matrix, rowvar=False)

    def get_history_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (min, max) across all stored vectors."""
        if not self._history:
            raise ValueError("No vectors in history")
        matrix = np.stack(self._history)
        return matrix.min(axis=0), matrix.max(axis=0)

    def get_all_vectors(self) -> np.ndarray:
        """Return all stored vectors as a matrix (n_samples x n_features)."""
        if not self._history:
            raise ValueError("No vectors in history")
        return np.stack(self._history)
