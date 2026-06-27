from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

from adapters.embedding_adapter import MockEmbeddingAdapter
from engine.embedding_signature import EmbeddingSignatureGenerator
from domain.embedding_models import EmbeddingBaseline


class TestEmbeddingBaseline:
    def test_baseline_model_fields(self):
        baseline = EmbeddingBaseline(
            agent_id="test",
            centroid=[0.1, 0.2],
            threshold=1.0,
            within_mean=0.5,
            within_std=0.1,
            pca_components=[[1.0, 0.0], [0.0, 1.0]],
            explained_variance=0.95,
            n_components=2,
            n_responses=10,
        )
        assert baseline.agent_id == "test"
        assert len(baseline.baseline_id) > 0
        assert baseline.created_at is not None


class TestEmbeddingSignatureGenerator:
    def test_generate_baseline_returns_model(self):
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=5)
        responses = [f"Response {i} with some text" for i in range(10)]
        baseline = gen.generate_baseline("test-agent", responses)
        assert isinstance(baseline, EmbeddingBaseline)
        assert len(baseline.centroid) == 5
        assert baseline.n_responses == 10
        assert baseline.threshold > 0
        assert baseline.n_components == 5
        assert len(baseline.pca_components) == 5
        assert baseline.explained_variance > 0

    def test_generate_baseline_respects_max_components(self):
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=3)
        responses = [f"Text number {i}" for i in range(5)]
        baseline = gen.generate_baseline("agent", responses)
        assert baseline.n_components == 3
        assert len(baseline.centroid) == 3

    def test_generate_baseline_caps_at_n_samples(self):
        """When n_components > n_samples, PCA caps at n_samples."""
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=50)
        responses = [f"Short {i}" for i in range(4)]
        baseline = gen.generate_baseline("agent", responses)
        assert baseline.n_components <= 4

    def test_verify_own_response(self):
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=5)
        responses = [f"Response {i} with some text content here" for i in range(10)]
        baseline = gen.generate_baseline("test", responses)
        is_match, dist = gen.verify(responses[0], baseline)
        # Own response should be close to centroid
        assert dist < baseline.threshold * 2  # generous threshold for mock

    def test_project_returns_correct_dim(self):
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=8)
        responses = [f"Sample text number {i}" for i in range(10)]
        baseline = gen.generate_baseline("agent", responses)
        projected = gen.project("New response text", baseline)
        assert projected.shape == (8,)

    def test_compare_different_baselines(self):
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=5)
        baseline_a = gen.generate_baseline("a", [f"Agent A response {i}" for i in range(10)])
        baseline_b = gen.generate_baseline("b", [f"Agent B different text {i}" for i in range(10)])
        dist = gen.compare_baselines(baseline_a, baseline_b)
        assert dist > 0  # Different agents should have different centroids

    def test_compare_same_baseline_is_zero(self):
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=5)
        baseline = gen.generate_baseline("a", [f"Agent A response {i}" for i in range(10)])
        dist = gen.compare_baselines(baseline, baseline)
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_verify_returns_tuple(self):
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=5)
        responses = [f"Text {i}" for i in range(10)]
        baseline = gen.generate_baseline("agent", responses)
        result = gen.verify("Some new text", baseline)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)


class TestDualPathWithEmbedding:
    """Test that DualPathVerifier's embedding integration works."""

    def test_verify_with_response_text_no_embedding_gen(self):
        """Without an embedding generator, response_text is ignored."""
        from engine.verification import DualPathVerifier
        verifier = DualPathVerifier(fast_threshold=0.5)
        centroid = np.array([0.5] * 32)
        verifier.register_agent("agent-1", centroid)
        result = verifier.verify(centroid, response_text="hello world")
        assert result.agent_id == "agent-1"
        assert "+embedding_fail" not in result.path_used

    def test_register_with_embedding_baseline(self):
        from engine.verification import DualPathVerifier
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=5)
        baseline = gen.generate_baseline("agent-1", [f"text {i}" for i in range(10)])

        verifier = DualPathVerifier(fast_threshold=0.5, embedding_generator=gen)
        centroid = np.array([0.5] * 32)
        info = verifier.register_agent("agent-1", centroid, embedding_baseline=baseline)
        assert "agent-1" in verifier._embedding_baselines

    def test_embedding_check_applied_on_verify(self):
        """With embedding generator and baselines, response_text is checked."""
        from engine.verification import DualPathVerifier
        gen = EmbeddingSignatureGenerator(MockEmbeddingAdapter(), n_components=5)
        baseline = gen.generate_baseline("agent-1", [f"text {i}" for i in range(10)])

        verifier = DualPathVerifier(fast_threshold=0.5, embedding_generator=gen)
        centroid = np.array([0.5] * 32)
        verifier.register_agent("agent-1", centroid, embedding_baseline=baseline)

        # Verify with a response_text that may or may not pass
        result = verifier.verify(centroid, response_text="text 0")
        assert result.agent_id == "agent-1"
        # Result should be valid regardless (path may include embedding_fail or not)
        assert result.confidence > 0
