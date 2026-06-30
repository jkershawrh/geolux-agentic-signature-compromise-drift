from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from sklearn.decomposition import PCA

from domain.embedding_models import EmbeddingBaseline
from engine.geometric.distance import euclidean_distance


def compute_eer(
    genuine_dists: np.ndarray, impostor_dists: np.ndarray
) -> tuple[float, float]:
    """Equal Error Rate from genuine/impostor distance arrays.

    Returns ``(eer, eer_threshold)`` where EER is the operating point at which
    FAR and FRR intersect (standard biometric definition).
    """
    if len(genuine_dists) == 0 or len(impostor_dists) == 0:
        return 0.5, 0.0

    max_dist = max(float(genuine_dists.max()), float(impostor_dists.max()))
    thresholds = np.linspace(0, max_dist * 1.2, 300)

    far_curve = np.array([float(np.mean(impostor_dists < t)) for t in thresholds])
    frr_curve = np.array([float(np.mean(genuine_dists > t)) for t in thresholds])

    diff = np.abs(far_curve - frr_curve)
    eer_idx = int(np.argmin(diff))
    eer = float((far_curve[eer_idx] + frr_curve[eer_idx]) / 2)
    return eer, float(thresholds[eer_idx])


@dataclass
class SharedEmbeddingSpace:
    """PCA fitted on ALL agents' embeddings so every centroid lives in one space."""

    components: np.ndarray  # (n_components, n_features)
    mean: np.ndarray  # (n_features,)
    n_components: int
    explained_variance: float

    @classmethod
    def fit(cls, embeddings: np.ndarray, n_components: int = 20) -> SharedEmbeddingSpace:
        if embeddings.ndim != 2 or embeddings.shape[0] < 2:
            raise ValueError("need at least two embeddings to fit shared PCA")
        n_comp = min(n_components, embeddings.shape[0] - 1, embeddings.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(embeddings)
        return cls(
            components=pca.components_,
            mean=pca.mean_,
            n_components=n_comp,
            explained_variance=float(sum(pca.explained_variance_ratio_)),
        )

    def transform(self, embedding: np.ndarray) -> np.ndarray:
        centered = np.asarray(embedding, dtype=np.float64) - self.mean
        return centered @ self.components.T

    def transform_batch(self, embeddings: np.ndarray) -> np.ndarray:
        centered = np.asarray(embeddings, dtype=np.float64) - self.mean
        return centered @ self.components.T


class EmbeddingSignatureGenerator:
    """Generate and verify embedding-space agent signatures.

    Identity verification requires a **shared** PCA space: fit once on the union
    of enrollment embeddings (``fit_shared``), then build per-agent baselines in
    that space. Per-agent PCA makes cross-agent distances meaningless.
    """

    def __init__(self, embedding_adapter, n_components: int = 20):
        self._adapter = embedding_adapter
        self._n_components = n_components
        self._shared_space: SharedEmbeddingSpace | None = None

    @property
    def shared_space(self) -> SharedEmbeddingSpace | None:
        return self._shared_space

    def fit_shared(
        self, data: Union[list[str], dict[str, list[str]]]
    ) -> SharedEmbeddingSpace:
        """Fit shared PCA on all provided responses (flat list or agent -> responses)."""
        if isinstance(data, dict):
            texts = [t for responses in data.values() for t in responses]
        else:
            texts = list(data)
        if len(texts) < 2:
            raise ValueError("fit_shared requires at least two responses")
        embeddings = np.array([self._adapter.embed(t) for t in texts])
        self._shared_space = SharedEmbeddingSpace.fit(embeddings, self._n_components)
        return self._shared_space

    def _space_for(self, responses: list[str]) -> SharedEmbeddingSpace:
        if self._shared_space is not None:
            return self._shared_space
        if len(responses) < 2:
            raise ValueError(
                "need fit_shared() with multiple agents, or at least two enrollment "
                "responses for single-agent PCA"
            )
        embeddings = np.array([self._adapter.embed(r) for r in responses])
        self._shared_space = SharedEmbeddingSpace.fit(embeddings, self._n_components)
        return self._shared_space

    def generate_baseline(self, agent_id: str, responses: list[str]) -> EmbeddingBaseline:
        """Build a PCA-reduced embedding baseline in the shared space."""
        if not responses:
            raise ValueError("need at least one response to build a baseline")
        space = self._space_for(responses)
        embeddings = np.array([self._adapter.embed(r) for r in responses])
        reduced = space.transform_batch(embeddings)

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
            pca_components=space.components.tolist(),
            pca_mean=space.mean.tolist(),
            explained_variance=space.explained_variance,
            n_components=space.n_components,
            n_responses=len(responses),
        )

    def project(self, response: str, baseline: EmbeddingBaseline) -> np.ndarray:
        """Project a response into the baseline's shared PCA space (mean-centered)."""
        return self.project_vector(self._adapter.embed(response), baseline)

    def project_vector(self, embedding: np.ndarray, baseline: EmbeddingBaseline) -> np.ndarray:
        """Project a pre-computed embedding vector using the baseline's PCA parameters."""
        mean = np.array(baseline.pca_mean, dtype=np.float64)
        components = np.array(baseline.pca_components, dtype=np.float64)
        return (np.asarray(embedding, dtype=np.float64) - mean) @ components.T

    def verify(self, response: str, baseline: EmbeddingBaseline) -> tuple[bool, float]:
        """Verify whether a response falls within the baseline's threshold."""
        projected = self.project(response, baseline)
        centroid = np.array(baseline.centroid)
        distance = euclidean_distance(projected, centroid)
        return distance <= baseline.threshold, float(distance)

    def compare_baselines(self, a: EmbeddingBaseline, b: EmbeddingBaseline) -> float:
        """Euclidean distance between two baselines' centroids (same PCA space required)."""
        if a.pca_mean != b.pca_mean or a.pca_components != b.pca_components:
            raise ValueError(
                "baselines were built in different PCA spaces; call fit_shared() "
                "on all agents before generate_baseline()"
            )
        return euclidean_distance(np.array(a.centroid), np.array(b.centroid))
