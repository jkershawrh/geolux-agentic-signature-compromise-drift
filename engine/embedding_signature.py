from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from domain.embedding_models import EmbeddingBaseline
from engine.geometric.distance import euclidean_distance


class EmbeddingSignatureGenerator:
    """Generate and verify embedding-space agent signatures.

    Embeds agent responses via an embedding adapter, reduces dimensionality
    with PCA, and builds a centroid-based baseline for identity verification.
    """

    def __init__(self, embedding_adapter, n_components: int = 20):
        self._adapter = embedding_adapter
        self._n_components = n_components

    def generate_baseline(self, agent_id: str, responses: list[str]) -> EmbeddingBaseline:
        """Build a PCA-reduced embedding baseline from a set of agent responses."""
        embeddings = np.array([self._adapter.embed(r) for r in responses])

        n_comp = min(self._n_components, len(responses), embeddings.shape[1])
        pca = PCA(n_components=n_comp)
        reduced = pca.fit_transform(embeddings)

        centroid = reduced.mean(axis=0)
        within_dists = [euclidean_distance(v, centroid) for v in reduced]
        within_mean = float(np.mean(within_dists))
        within_std = float(np.std(within_dists)) + 1e-10
        threshold = within_mean + 2 * within_std

        return EmbeddingBaseline(
            agent_id=agent_id,
            centroid=centroid.tolist(),
            threshold=threshold,
            within_mean=within_mean,
            within_std=within_std,
            pca_components=pca.components_.tolist(),
            explained_variance=float(sum(pca.explained_variance_ratio_)),
            n_components=n_comp,
            n_responses=len(responses),
        )

    def project(self, response: str, baseline: EmbeddingBaseline) -> np.ndarray:
        """Project a response into the baseline's PCA space."""
        embedding = self._adapter.embed(response)
        components = np.array(baseline.pca_components)
        # Manual PCA transform: project embedding onto learned components
        projected = embedding @ components.T
        return projected

    def verify(self, response: str, baseline: EmbeddingBaseline) -> tuple[bool, float]:
        """Verify whether a response falls within the baseline's threshold."""
        projected = self.project(response, baseline)
        centroid = np.array(baseline.centroid)
        distance = euclidean_distance(projected, centroid)
        return distance <= baseline.threshold, float(distance)

    def compare_baselines(self, a: EmbeddingBaseline, b: EmbeddingBaseline) -> float:
        """Euclidean distance between two baselines' centroids."""
        return euclidean_distance(np.array(a.centroid), np.array(b.centroid))
