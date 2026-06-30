from __future__ import annotations

from typing import Optional

import numpy as np


def reduce_to_manifold(vectors: np.ndarray,
                       n_components: int = 2,
                       method: str = "umap",
                       random_state: int = 42) -> np.ndarray:
    """Project high-dimensional metric vectors onto a low-dimensional manifold.

    Supports UMAP (default) and PCA as reduction methods.
    UMAP preserves both local and global structure, making it better for
    detecting subtle behavioral drift. PCA is faster and deterministic.

    Returns an array of shape (n_samples, n_components).
    """
    if vectors.shape[0] < 2:
        raise ValueError("Need at least 2 vectors for manifold reduction")

    if method == "pca":
        return _pca_reduce(vectors, n_components)
    elif method == "umap":
        if vectors.shape[0] < 3:
            return _pca_reduce(vectors, n_components)
        return _umap_reduce(vectors, n_components, random_state)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'umap' or 'pca'.")


def _pca_reduce(vectors: np.ndarray, n_components: int) -> np.ndarray:
    """PCA dimensionality reduction."""
    from sklearn.decomposition import PCA

    n_components = min(n_components, vectors.shape[0], vectors.shape[1])
    pca = PCA(n_components=n_components)
    return pca.fit_transform(vectors)


def _umap_reduce(vectors: np.ndarray, n_components: int,
                 random_state: int) -> np.ndarray:
    """UMAP dimensionality reduction."""
    import umap

    n_neighbors = min(15, vectors.shape[0] - 1)
    n_neighbors = max(2, n_neighbors)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    )
    return reducer.fit_transform(vectors)


def project_point(point: np.ndarray, reference_vectors: np.ndarray,
                  reference_projections: np.ndarray,
                  k_neighbors: int = 5) -> np.ndarray:
    """Project a new point onto an existing manifold using k-NN interpolation.

    This avoids re-fitting the entire manifold for each new observation.
    Uses weighted average of the k nearest reference projections.
    """
    distances = np.linalg.norm(reference_vectors - point, axis=1)
    k = min(k_neighbors, len(reference_vectors))
    nearest_idx = np.argsort(distances)[:k]

    nearest_distances = distances[nearest_idx]
    nearest_projections = reference_projections[nearest_idx]

    # Inverse distance weighting
    weights = 1.0 / (nearest_distances + 1e-10)
    weights /= weights.sum()

    return np.average(nearest_projections, axis=0, weights=weights)
