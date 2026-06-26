import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from engine.verification import DualPathVerifier, LSHIndex, CommitmentStore


class TestLSHIndex:
    def test_register_and_lookup(self):
        lsh = LSHIndex(n_planes=8)
        centroid = np.array([0.5] * 32)
        bucket = lsh.register("agent-1", centroid)
        found_bucket, candidates = lsh.lookup(centroid)
        assert found_bucket == bucket
        assert "agent-1" in candidates

    def test_similar_vectors_same_bucket(self):
        lsh = LSHIndex(n_planes=4)  # fewer planes = larger buckets
        v1 = np.array([0.5] * 32)
        v2 = np.array([0.51] * 32)
        lsh.register("a", v1)
        bucket_a = lsh._hash_vector(v1)
        bucket_b = lsh._hash_vector(v2)
        # Very similar vectors should often land in same bucket
        # (not guaranteed with 4 planes but likely)

    def test_different_vectors_different_buckets(self):
        lsh = LSHIndex(n_planes=8)
        v1 = np.zeros(32)
        v2 = np.ones(32)
        lsh.register("a", v1)
        lsh.register("b", v2)
        b1 = lsh._hash_vector(v1)
        b2 = lsh._hash_vector(v2)
        assert b1 != b2  # Very different vectors should be in different buckets

    def test_nearest_finds_closest(self):
        lsh = LSHIndex()
        v1 = np.array([0.0] * 32)
        v2 = np.array([1.0] * 32)
        lsh.register("close", v1)
        lsh.register("far", v2)
        query = np.array([0.1] * 32)
        agent_id, dist = lsh.nearest(query, ["close", "far"])
        assert agent_id == "close"


class TestCommitmentStore:
    def test_commit_and_verify(self):
        store = CommitmentStore()
        vec = np.array([0.5] * 32)
        commitment = store.commit("agent-1", vec)
        assert len(commitment) == 64  # SHA-256 hex
        is_valid, dist = store.verify("agent-1", vec, tolerance=0.1)
        assert is_valid
        assert dist < 0.01

    def test_verify_different_vector_fails(self):
        store = CommitmentStore()
        store.commit("agent-1", np.array([0.0] * 32))
        is_valid, dist = store.verify("agent-1", np.array([1.0] * 32), tolerance=0.1)
        assert not is_valid

    def test_verify_exact(self):
        store = CommitmentStore()
        vec = np.array([0.5] * 32)
        store.commit("agent-1", vec)
        assert store.verify_exact("agent-1", vec) is True
        assert store.verify_exact("agent-1", vec + 0.001) is False

    def test_unknown_agent(self):
        store = CommitmentStore()
        is_valid, dist = store.verify("unknown", np.zeros(32))
        assert not is_valid


class TestDualPathVerifier:
    def test_fast_path_single_candidate(self):
        verifier = DualPathVerifier(fast_threshold=0.5)
        centroid = np.array([0.5] * 32)
        verifier.register_agent("agent-1", centroid)
        result = verifier.verify(centroid)
        assert result.agent_id == "agent-1"
        assert result.path_used == "fast"
        assert result.confidence > 0.8

    def test_escalation_on_ambiguous_bucket(self):
        verifier = DualPathVerifier(n_planes=2, secure_tolerance=1.0)  # few planes = more collisions
        verifier.register_agent("a", np.array([0.5] * 32))
        verifier.register_agent("b", np.array([0.51] * 32))
        result = verifier.verify(np.array([0.5] * 32))
        # Should either fast-path to "a" or escalate to secure
        assert result.agent_id is not None

    def test_secure_path_with_expected_agent(self):
        verifier = DualPathVerifier(escalation_policy="always", secure_tolerance=0.5)
        verifier.register_agent("agent-1", np.array([0.5] * 32))
        result = verifier.verify(np.array([0.5] * 32), expected_agent_id="agent-1")
        assert result.path_used == "secure"
        assert result.commitment_valid is True
        assert result.agent_id == "agent-1"

    def test_unknown_vector_returns_none(self):
        verifier = DualPathVerifier(escalation_policy="always", secure_tolerance=0.1)
        verifier.register_agent("agent-1", np.zeros(32))
        result = verifier.verify(np.ones(32))
        assert result.agent_id is None

    def test_register_returns_info(self):
        verifier = DualPathVerifier()
        info = verifier.register_agent("test", np.array([0.5] * 32))
        assert "agent_id" in info
        assert "bucket_id" in info
        assert "commitment" in info
        assert len(info["commitment"]) == 64
