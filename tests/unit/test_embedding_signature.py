from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from sklearn.decomposition import PCA

from adapters.embedding_adapter import MockEmbeddingAdapter
from domain.embedding_models import EmbeddingBaseline
from engine.embedding_signature import (
    EmbeddingSignatureGenerator,
    SharedEmbeddingSpace,
    compute_eer,
)


class TestEmbeddingBaseline:
    def test_baseline_model_fields(self):
        baseline = EmbeddingBaseline(
            agent_id="test",
            centroid=[0.1, 0.2],
            threshold=1.0,
            within_mean=0.5,
            within_std=0.1,
            pca_components=[[1.0, 0.0], [0.0, 1.0]],
            pca_mean=[0.0, 0.0],
            explained_variance=0.95,
            n_components=2,
            n_responses=10,
        )
        assert baseline.agent_id == "test"
        assert len(baseline.baseline_id) > 0
        assert baseline.created_at is not None


class TestComputeEer:
    def test_perfect_separation(self):
        genuine = np.array([0.1, 0.2, 0.15])
        impostor = np.array([2.0, 2.5, 2.2])
        eer, _ = compute_eer(genuine, impostor)
        assert eer < 0.05

    def test_empty_returns_chance(self):
        eer, _ = compute_eer(np.array([]), np.array([1.0]))
        assert eer == 0.5


class TestSharedEmbeddingSpace:
    def test_transform_matches_sklearn(self):
        rng = np.random.default_rng(0)
        embeddings = rng.standard_normal((20, 64))
        space = SharedEmbeddingSpace.fit(embeddings, n_components=5)
        pca = PCA(n_components=5)
        pca.fit(embeddings)
        for row in embeddings:
            assert space.transform(row) == pytest.approx(pca.transform(row.reshape(1, -1))[0], abs=1e-10)


class TestEmbeddingSignatureGenerator:
    def test_generate_baseline_returns_model(self):
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=5)
        responses = [f"Response {i} with some text" for i in range(10)]
        gen.fit_shared(responses)
        baseline = gen.generate_baseline("test-agent", responses)
        assert isinstance(baseline, EmbeddingBaseline)
        assert len(baseline.centroid) == 5
        assert len(baseline.pca_mean) == 768
        assert baseline.n_responses == 10
        assert baseline.threshold > 0

    def test_project_matches_sklearn_on_enrollment(self):
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=5)
        responses = [f"Response {i} with some text" for i in range(10)]
        gen.fit_shared(responses)
        baseline = gen.generate_baseline("test", responses)
        for text in responses:
            proj = gen.project(text, baseline)
            assert proj.shape == (baseline.n_components,)

    def test_verify_own_response(self):
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=5)
        responses = [f"Response {i} with some text content here" for i in range(10)]
        gen.fit_shared(responses)
        baseline = gen.generate_baseline("test", responses)
        is_match, dist = gen.verify(responses[0], baseline)
        assert is_match
        assert dist <= baseline.threshold

    def test_compare_baselines_requires_shared_space(self):
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=5)
        resp_a = [f"Agent A response {i} unique alpha text" for i in range(10)]
        resp_b = [f"Agent B response {i} unique beta text" for i in range(10)]
        gen.fit_shared({"a": resp_a, "b": resp_b})
        baseline_a = gen.generate_baseline("a", resp_a)
        baseline_b = gen.generate_baseline("b", resp_b)
        dist = gen.compare_baselines(baseline_a, baseline_b)
        assert dist > 0

    def test_compare_rejects_different_pca_spaces(self):
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=5)
        gen.fit_shared([f"a{i}" for i in range(10)])
        ba = gen.generate_baseline("a", [f"a{i}" for i in range(10)])
        gen.fit_shared([f"b{i}" for i in range(10)])
        bb = gen.generate_baseline("b", [f"b{i}" for i in range(10)])
        with pytest.raises(ValueError, match="different PCA spaces"):
            gen.compare_baselines(ba, bb)

    def test_project_is_mean_centered_not_raw_dot_product(self):
        """Regression: old project() skipped PCA mean subtraction."""
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=5)
        responses = [f"Response {i} with text" for i in range(10)]
        gen.fit_shared(responses)
        baseline = gen.generate_baseline("t", responses)
        space = gen.shared_space
        emb = gen._adapter.embed(responses[0])
        correct = space.transform(emb)
        wrong = emb @ np.array(baseline.pca_components).T  # pre-fix bug
        assert gen.project(responses[0], baseline) == pytest.approx(correct, abs=1e-10)
        assert not np.allclose(correct, wrong, atol=1e-6)

    def test_shared_pca_same_mean_across_agents(self):
        """Regression: per-agent PCA put centroids in incomparable spaces."""
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=5)
        resp_a = [f"Agent A unique {i}" for i in range(10)]
        resp_b = [f"Agent B unique {i}" for i in range(10)]
        gen.fit_shared({"a": resp_a, "b": resp_b})
        ba = gen.generate_baseline("a", resp_a)
        bb = gen.generate_baseline("b", resp_b)
        assert ba.pca_mean == bb.pca_mean
        assert ba.pca_components == bb.pca_components
        gen.compare_baselines(ba, bb)  # must not raise


class TestDualPathWithEmbedding:
    def test_register_with_embedding_baseline(self):
        from engine.verification import DualPathVerifier

        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=5)
        responses = [f"text {i}" for i in range(10)]
        gen.fit_shared(responses)
        baseline = gen.generate_baseline("agent-1", responses)

        verifier = DualPathVerifier(fast_threshold=0.5, embedding_generator=gen)
        centroid = np.array([0.5] * 32)
        verifier.register_agent("agent-1", centroid, embedding_baseline=baseline)
        assert "agent-1" in verifier._embedding_baselines

    def test_embedding_check_applied_on_verify(self):
        from engine.verification import DualPathVerifier

        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=5)
        responses = [f"text {i}" for i in range(10)]
        gen.fit_shared(responses)
        baseline = gen.generate_baseline("agent-1", responses)

        verifier = DualPathVerifier(fast_threshold=0.5, embedding_generator=gen)
        centroid = np.array([0.5] * 32)
        verifier.register_agent("agent-1", centroid, embedding_baseline=baseline)

        result = verifier.verify(centroid, response_text="text 0")
        assert result.agent_id == "agent-1"
        assert result.confidence > 0
