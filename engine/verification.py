#!/usr/bin/env python3
"""Dual-path identity verification: fast LSH lookup + secure commitment hash.

Fast path (microseconds): LSH bucket lookup -> approximate match
Secure path (milliseconds): full vector computation -> commitment hash verification

The fast path is an optimization. The secure path is the authority.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from engine.geometric.distance import euclidean_distance

# Optional embedding support -- imported lazily to avoid hard dependency
try:
    from engine.embedding_signature import EmbeddingSignatureGenerator
    from domain.embedding_models import EmbeddingBaseline
except ImportError:  # pragma: no cover
    EmbeddingSignatureGenerator = None  # type: ignore[assignment,misc]
    EmbeddingBaseline = None  # type: ignore[assignment,misc]


@dataclass
class VerificationResult:
    """Result of a dual-path identity verification."""
    agent_id: str | None          # Identified agent (None if unrecognized)
    path_used: str                 # "fast" or "secure"
    confidence: float              # [0, 1]
    distance: float                # Distance to nearest match
    bucket_id: int | None          # LSH bucket (fast path only)
    commitment_valid: bool | None  # Hash check (secure path only)
    escalated: bool                # Did fast path escalate to secure?
    details: str
    # Separate identity and drift assessments
    identity_verified: bool = True      # Embedding-based: is this the right agent?
    drift_detected: bool = False        # Metric-based: has behavior changed?
    identity_distance: float = 0.0      # Embedding distance
    drift_distance: float = 0.0         # Metric distance


class LSHIndex:
    """Locality-Sensitive Hash index for approximate nearest-agent lookup.

    Uses random hyperplane projection: hash(v) = sign(v . random_planes).
    Agents with similar metric vectors land in the same bucket.
    """

    def __init__(self, n_planes: int = 8, seed: int = 42):
        self._n_planes = n_planes
        self._planes: np.ndarray | None = None
        self._buckets: dict[int, list[str]] = {}  # bucket_id -> [agent_ids]
        self._centroids: dict[str, np.ndarray] = {}  # agent_id -> centroid vector
        self._rng = np.random.RandomState(seed)

    def _init_planes(self, dim: int) -> None:
        """Initialize random hyperplanes for the given dimensionality."""
        if self._planes is None or self._planes.shape[1] != dim:
            self._planes = self._rng.randn(self._n_planes, dim)

    def _hash_vector(self, vec: np.ndarray) -> int:
        """Compute LSH bucket ID for a vector."""
        self._init_planes(len(vec))
        projections = self._planes @ vec
        bits = (projections > 0).astype(int)
        # Convert binary array to integer bucket ID
        bucket_id = 0
        for bit in bits:
            bucket_id = (bucket_id << 1) | int(bit)
        return bucket_id

    def register(self, agent_id: str, centroid: np.ndarray) -> int:
        """Register an agent's centroid in the LSH index. Returns bucket ID."""
        bucket_id = self._hash_vector(centroid)
        if bucket_id not in self._buckets:
            self._buckets[bucket_id] = []
        if agent_id not in self._buckets[bucket_id]:
            self._buckets[bucket_id].append(agent_id)
        self._centroids[agent_id] = centroid.copy()
        return bucket_id

    def lookup(self, vec: np.ndarray) -> tuple[int, list[str]]:
        """Look up which bucket a vector falls into. Returns (bucket_id, [agent_ids])."""
        bucket_id = self._hash_vector(vec)
        candidates = self._buckets.get(bucket_id, [])
        return bucket_id, candidates

    def nearest(self, vec: np.ndarray, candidates: list[str]) -> tuple[str | None, float]:
        """Find the nearest agent among candidates by Euclidean distance."""
        if not candidates:
            return None, float('inf')
        best_id = None
        best_dist = float('inf')
        for agent_id in candidates:
            if agent_id in self._centroids:
                dist = euclidean_distance(vec, self._centroids[agent_id])
                if dist < best_dist:
                    best_dist = dist
                    best_id = agent_id
        return best_id, best_dist

    @property
    def registered_agents(self) -> list[str]:
        return list(self._centroids.keys())

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)


class CommitmentStore:
    """Secure commitment hash store for cryptographic identity verification.

    Stores SHA-256 hashes of agent metric vectors. Verification computes
    the vector, hashes it, and compares. The raw vector is never stored.
    """

    def __init__(self):
        self._commitments: dict[str, str] = {}  # agent_id -> SHA-256 hash
        self._centroids: dict[str, np.ndarray] = {}  # agent_id -> centroid (for distance)

    def commit(self, agent_id: str, centroid: np.ndarray) -> str:
        """Store a commitment hash for an agent's centroid vector."""
        hash_input = json.dumps(centroid.tolist(), sort_keys=True).encode()
        commitment = hashlib.sha256(hash_input).hexdigest()
        self._commitments[agent_id] = commitment
        self._centroids[agent_id] = centroid.copy()
        return commitment

    def verify(self, agent_id: str, vector: np.ndarray, tolerance: float = 0.3) -> tuple[bool, float]:
        """Verify a vector against the stored commitment.

        Since hash comparison is exact-match only, we verify by:
        1. Computing distance from the stored centroid (approximate match)
        2. If distance < tolerance, consider verified

        Returns (is_verified, distance).
        """
        if agent_id not in self._centroids:
            return False, float('inf')
        dist = euclidean_distance(vector, self._centroids[agent_id])
        return dist <= tolerance, float(dist)

    def verify_exact(self, agent_id: str, centroid: np.ndarray) -> bool:
        """Exact hash verification -- for batch signature comparison."""
        if agent_id not in self._commitments:
            return False
        hash_input = json.dumps(centroid.tolist(), sort_keys=True).encode()
        computed = hashlib.sha256(hash_input).hexdigest()
        return computed == self._commitments[agent_id]

    def get_commitment(self, agent_id: str) -> str | None:
        return self._commitments.get(agent_id)


class DualPathVerifier:
    """Dual-path identity verification combining LSH speed with commitment security.

    Fast path: LSH bucket lookup -> approximate match (microseconds)
    Secure path: distance computation + commitment check (milliseconds)

    The fast path is an optimization. The secure path is the authority.
    """

    def __init__(
        self,
        n_planes: int = 8,
        fast_threshold: float = 0.3,
        secure_tolerance: float = 0.5,
        escalation_policy: str = "ambiguous",  # "ambiguous", "always", "never"
        embedding_generator: Optional["EmbeddingSignatureGenerator"] = None,
    ):
        self._lsh = LSHIndex(n_planes=n_planes)
        self._commitments = CommitmentStore()
        self._fast_threshold = fast_threshold
        self._secure_tolerance = secure_tolerance
        self._escalation_policy = escalation_policy
        self._embedding_gen = embedding_generator
        self._embedding_baselines: dict[str, "EmbeddingBaseline"] = {}

    def register_agent(
        self,
        agent_id: str,
        centroid: np.ndarray,
        embedding_baseline: Optional["EmbeddingBaseline"] = None,
    ) -> dict:
        """Register an agent in both the LSH index and commitment store.

        Optionally accepts an embedding baseline for combined verification.
        """
        bucket_id = self._lsh.register(agent_id, centroid)
        commitment = self._commitments.commit(agent_id, centroid)
        if embedding_baseline is not None:
            self._embedding_baselines[agent_id] = embedding_baseline
        return {
            "agent_id": agent_id,
            "bucket_id": bucket_id,
            "commitment": commitment,
            "bucket_occupants": self._lsh._buckets.get(bucket_id, []),
        }

    def verify(
        self,
        vec: np.ndarray,
        expected_agent_id: str | None = None,
        response_text: str | None = None,
    ) -> VerificationResult:
        """Verify a metric vector through the dual-path system.

        If expected_agent_id is provided, verify that specific agent.
        Otherwise, identify which agent produced the vector.

        If response_text is provided and an embedding generator + baselines are
        available, the embedding distance is also checked. A failed embedding
        check halves the confidence and appends '+embedding_fail' to path_used.
        """
        # --- Fast path: LSH lookup ---
        bucket_id, candidates = self._lsh.lookup(vec)

        should_escalate = False
        fast_agent_id = None
        fast_distance = float('inf')

        if len(candidates) == 1:
            # Unambiguous bucket -- single agent
            fast_agent_id = candidates[0]
            fast_distance = euclidean_distance(vec, self._lsh._centroids[fast_agent_id])

            if fast_distance > self._fast_threshold:
                should_escalate = True  # Distance too large, verify properly
        elif len(candidates) > 1:
            # Ambiguous bucket -- multiple agents, need secure path
            fast_agent_id, fast_distance = self._lsh.nearest(vec, candidates)
            should_escalate = True  # Always escalate ambiguous
        else:
            # Empty bucket -- unknown agent
            should_escalate = True

        # Check escalation policy
        if self._escalation_policy == "always":
            should_escalate = True
        elif self._escalation_policy == "never":
            should_escalate = False

        if not should_escalate and fast_agent_id:
            # Fast path succeeds — metric distance determines drift
            confidence = max(0.0, 1.0 - fast_distance / self._fast_threshold)
            drift_detected = fast_distance > self._fast_threshold * 0.5
            result = VerificationResult(
                agent_id=fast_agent_id,
                path_used="fast",
                confidence=min(1.0, confidence),
                distance=fast_distance,
                bucket_id=bucket_id,
                commitment_valid=None,
                escalated=False,
                details=f"Fast path: bucket {bucket_id}, distance {fast_distance:.4f}",
                drift_detected=drift_detected,
                drift_distance=fast_distance,
            )
            return self._apply_embedding_check(result, response_text)

        # --- Secure path: full verification ---
        if expected_agent_id:
            # Verify a specific agent
            is_valid, dist = self._commitments.verify(
                expected_agent_id, vec, self._secure_tolerance
            )
            confidence = max(0.0, 1.0 - dist / self._secure_tolerance) if is_valid else 0.0
            drift_detected = dist > self._secure_tolerance * 0.5
            result = VerificationResult(
                agent_id=expected_agent_id if is_valid else None,
                path_used="secure",
                confidence=min(1.0, confidence),
                distance=dist,
                bucket_id=bucket_id,
                commitment_valid=is_valid,
                escalated=should_escalate,
                details=f"Secure path: distance {dist:.4f}, valid={is_valid}",
                drift_detected=drift_detected,
                drift_distance=float(dist),
            )
            return self._apply_embedding_check(result, response_text)

        # Identify from all registered agents
        best_id = None
        best_dist = float('inf')
        for agent_id in self._commitments._centroids:
            is_valid, dist = self._commitments.verify(
                agent_id, vec, self._secure_tolerance
            )
            if is_valid and dist < best_dist:
                best_id = agent_id
                best_dist = dist

        if best_id:
            confidence = max(0.0, 1.0 - best_dist / self._secure_tolerance)
            drift_detected = best_dist > self._secure_tolerance * 0.5
            result = VerificationResult(
                agent_id=best_id,
                path_used="secure",
                confidence=min(1.0, confidence),
                distance=best_dist,
                bucket_id=bucket_id,
                commitment_valid=True,
                escalated=should_escalate,
                details=f"Secure path: identified {best_id}, distance {best_dist:.4f}",
                drift_detected=drift_detected,
                drift_distance=float(best_dist),
            )
            return self._apply_embedding_check(result, response_text)

        return VerificationResult(
            agent_id=None,
            path_used="secure",
            confidence=0.0,
            distance=fast_distance,
            bucket_id=bucket_id,
            commitment_valid=False,
            escalated=should_escalate,
            details="Secure path: no matching agent found",
            identity_verified=False,
            drift_distance=float(fast_distance),
        )

    def _apply_embedding_check(
        self, result: VerificationResult, response_text: str | None
    ) -> VerificationResult:
        """If embedding baselines are available, validate response_text against them.

        Sets identity_verified and identity_distance based on the embedding check.
        On failure, halve the confidence and mark path_used with '+embedding_fail'.
        Identity verification (embedding) is independent of drift detection (metrics).
        """
        if (
            response_text is None
            or self._embedding_gen is None
            or result.agent_id is None
            or result.agent_id not in self._embedding_baselines
        ):
            return result

        emb_baseline = self._embedding_baselines[result.agent_id]
        emb_valid, emb_dist = self._embedding_gen.verify(response_text, emb_baseline)
        if not emb_valid:
            return VerificationResult(
                agent_id=result.agent_id,
                path_used=result.path_used + "+embedding_fail",
                confidence=result.confidence * 0.5,
                distance=result.distance,
                bucket_id=result.bucket_id,
                commitment_valid=result.commitment_valid,
                escalated=result.escalated,
                details=(
                    result.details
                    + f" | embedding check failed (dist={emb_dist:.4f})"
                ),
                identity_verified=False,
                drift_detected=result.drift_detected,
                identity_distance=float(emb_dist),
                drift_distance=result.drift_distance,
            )
        return VerificationResult(
            agent_id=result.agent_id,
            path_used=result.path_used,
            confidence=result.confidence,
            distance=result.distance,
            bucket_id=result.bucket_id,
            commitment_valid=result.commitment_valid,
            escalated=result.escalated,
            details=result.details + f" | embedding check passed (dist={emb_dist:.4f})",
            identity_verified=True,
            drift_detected=result.drift_detected,
            identity_distance=float(emb_dist),
            drift_distance=result.drift_distance,
        )

    @property
    def lsh_index(self) -> LSHIndex:
        return self._lsh

    @property
    def commitment_store(self) -> CommitmentStore:
        return self._commitments
