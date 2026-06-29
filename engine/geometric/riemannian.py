from __future__ import annotations

import numpy as np


def compute_metric_tensor(covariance: np.ndarray,
                          regularization: float = 1e-6) -> np.ndarray:
    """Compute a Riemannian metric tensor from a covariance matrix.

    The metric tensor is the inverse of the covariance matrix (precision matrix).
    This encodes how "important" each direction is: directions with low variance
    (high precision) get amplified, making drift along those directions more
    significant. This is the Fisher information metric interpretation.

    Regularization prevents singularity when some metrics have zero variance.
    """
    n = covariance.shape[0]
    regularized = covariance + regularization * np.eye(n)
    return np.linalg.inv(regularized)


def compute_metric_tensor_shrinkage(vectors: np.ndarray,
                                    regularization: float = 1e-6) -> np.ndarray:
    """Compute metric tensor using Ledoit-Wolf shrinkage.

    Better than cov + epsilon*I when n_samples << n_features.
    """
    from sklearn.covariance import LedoitWolf
    if vectors.shape[0] < 2:
        return np.eye(vectors.shape[1])
    lw = LedoitWolf().fit(vectors)
    cov = lw.covariance_
    return np.linalg.inv(cov + regularization * np.eye(cov.shape[0]))


def local_metric_tensor(vectors: np.ndarray, point: np.ndarray,
                        k_neighbors: int = 10,
                        regularization: float = 1e-6) -> np.ndarray:
    """Compute a local Riemannian metric tensor at a specific point.

    Uses k-nearest neighbors to estimate the local covariance structure,
    following the approach from Riemannian-Geometric Fingerprints (arxiv 2506.22802).
    The local metric tensor captures how the geometry varies across the manifold.
    """
    distances = np.linalg.norm(vectors - point, axis=1)
    k = min(k_neighbors, len(vectors))
    nearest_idx = np.argsort(distances)[:k]
    neighbors = vectors[nearest_idx]

    local_cov = np.cov(neighbors, rowvar=False)
    if local_cov.ndim == 0:
        local_cov = np.array([[local_cov]])

    return compute_metric_tensor(local_cov, regularization)


def parallel_transport(vector: np.ndarray,
                       metric_at_source: np.ndarray,
                       metric_at_target: np.ndarray) -> np.ndarray:
    """Approximate parallel transport of a tangent vector between two points.

    Uses the Schild's ladder approximation for manifolds with varying metric.
    For small distances, this transforms a vector from one tangent space
    to another accounting for curvature.

    Experimental: approximate parallel transport. Not used in production pipeline.
    """
    # Compute transformation: sqrt(G_target^{-1} @ G_source)
    try:
        target_inv = np.linalg.inv(metric_at_target)
        transform = np.linalg.cholesky(target_inv @ metric_at_source)
        return transform @ vector
    except np.linalg.LinAlgError:
        return vector


def anisotropy_estimate(metric_tensor: np.ndarray,
                        perturbation_scale: float = 0.01) -> float:
    """Estimates anisotropy of the metric tensor from eigenvalue spread.

    Higher values indicate more directional sensitivity. Not a true
    sectional curvature computation.
    """
    eigenvalues = np.linalg.eigvalsh(metric_tensor)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) < 2:
        return 0.0
    log_eigenvalues = np.log(eigenvalues)
    return float(np.var(log_eigenvalues))
