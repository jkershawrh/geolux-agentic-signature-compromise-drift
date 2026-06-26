from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from domain.geometry import GeometricSignature
from engine.geometric.distance import (
    cosine_similarity,
    euclidean_distance,
    geodesic_distance,
)


@dataclass
class AuthenticationResult:
    """Result of verifying an agent's identity against its baseline signature."""

    is_authentic: bool
    confidence: float
    geodesic_distance: float
    euclidean_distance: float
    cosine_similarity: float
    threshold_used: float
    details: str


class AuthenticationEngine:
    """Verify agent identity by comparing current signature to stored baseline.

    Uses Euclidean distance for authentication (identity verification) to avoid
    the Riemannian metric tensor amplifying noise in low-variance dimensions.
    Geodesic distance is still computed and reported for drift detection.
    """

    def __init__(
        self,
        distance_threshold: float = 0.5,
        cosine_threshold: float = 0.85,
    ):
        self._distance_threshold = distance_threshold
        self._cosine_threshold = cosine_threshold

    def verify(
        self,
        current_signature: GeometricSignature,
        baseline_signature: GeometricSignature,
    ) -> AuthenticationResult:
        """Compare a current signature against a baseline to verify identity."""
        current_vec = np.array(current_signature.embedding_vector)
        baseline_vec = np.array(baseline_signature.embedding_vector)

        metric_tensor = None
        if baseline_signature.metric_tensor is not None:
            metric_tensor = np.array(baseline_signature.metric_tensor)

        geo_dist = geodesic_distance(current_vec, baseline_vec, metric_tensor)
        euc_dist = euclidean_distance(current_vec, baseline_vec)
        cos_sim = cosine_similarity(current_vec, baseline_vec)

        # Use Euclidean distance for authentication decisions — the Riemannian
        # metric tensor amplifies small differences in low-variance dimensions,
        # causing same-agent verification to fail.  Geodesic distance is still
        # reported for downstream drift detection.
        distance_ok = euc_dist <= self._distance_threshold
        cosine_ok = cos_sim >= self._cosine_threshold

        is_authentic = distance_ok and cosine_ok

        confidence = self._compute_confidence(euc_dist, cos_sim)

        if is_authentic:
            details = (
                f"Agent verified: euclidean_distance={euc_dist:.4f} "
                f"(<= {self._distance_threshold}), "
                f"cosine_similarity={cos_sim:.4f} "
                f"(>= {self._cosine_threshold}), "
                f"geodesic_distance={geo_dist:.4f}"
            )
        else:
            reasons = []
            if not distance_ok:
                reasons.append(
                    f"euclidean_distance={euc_dist:.4f} "
                    f"exceeds threshold {self._distance_threshold}"
                )
            if not cosine_ok:
                reasons.append(
                    f"cosine_similarity={cos_sim:.4f} "
                    f"below threshold {self._cosine_threshold}"
                )
            details = f"Agent NOT verified: {'; '.join(reasons)}"

        return AuthenticationResult(
            is_authentic=is_authentic,
            confidence=confidence,
            geodesic_distance=geo_dist,
            euclidean_distance=euc_dist,
            cosine_similarity=cos_sim,
            threshold_used=self._distance_threshold,
            details=details,
        )

    def _compute_confidence(self, euc_dist: float, cos_sim: float) -> float:
        """Compute a confidence score [0,1] combining distance and similarity.

        Uses Euclidean distance (not geodesic) so the confidence tracks the
        same metric used for the is_authentic decision.
        """
        distance_score = float(np.exp(-euc_dist / self._distance_threshold))
        cosine_score = max(0.0, (cos_sim - self._cosine_threshold) / (1.0 - self._cosine_threshold))
        cosine_score = min(1.0, cosine_score)

        confidence = 0.6 * distance_score + 0.4 * cosine_score
        return max(0.0, min(1.0, confidence))

    def identify_agent(
        self,
        current_signature: GeometricSignature,
        candidate_baselines: list[GeometricSignature],
    ) -> tuple[Optional[str], AuthenticationResult]:
        """Identify which agent produced a signature from a list of candidates.

        Returns (agent_id, best_result) or (None, best_result) if no match.
        """
        best_result: Optional[AuthenticationResult] = None
        best_agent_id: Optional[str] = None

        for baseline in candidate_baselines:
            result = self.verify(current_signature, baseline)
            if best_result is None or result.confidence > best_result.confidence:
                best_result = result
                best_agent_id = baseline.agent_id

        if best_result is None:
            return None, AuthenticationResult(
                is_authentic=False,
                confidence=0.0,
                geodesic_distance=float("inf"),
                euclidean_distance=float("inf"),
                cosine_similarity=0.0,
                threshold_used=self._distance_threshold,
                details="No candidate baselines provided",
            )

        if not best_result.is_authentic:
            return None, best_result

        return best_agent_id, best_result
