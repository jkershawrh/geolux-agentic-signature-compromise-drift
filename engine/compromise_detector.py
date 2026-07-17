from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from domain.enums import DriftCategory
from domain.geometry import DriftMeasurement


@dataclass
class CompromiseAlert:
    """Alert generated when potential compromise is detected."""
    agent_id: str
    severity: str  # "warning" or "critical"
    drift_category: DriftCategory
    compromise_probability: float
    geodesic_distance: float
    drift_magnitude: float
    evidence: dict
    recommendation: str


class CompromiseDetector:
    """Detect agent compromise from drift measurements.

    Uses threshold-based detection with configurable warning and critical levels.
    """

    def __init__(
        self,
        warning_threshold: float = 0.5,
        critical_threshold: float = 0.8,
        distance_warning: float = 0.5,
        distance_critical: float = 1.0,
    ):
        self._warning_threshold = warning_threshold
        self._critical_threshold = critical_threshold
        self._distance_warning = distance_warning
        self._distance_critical = distance_critical

    def evaluate(self, drift: DriftMeasurement) -> Optional[CompromiseAlert]:
        """Evaluate a drift measurement for signs of compromise.

        Returns a CompromiseAlert if thresholds are exceeded, else None.
        """
        if not drift.is_significant:
            if drift.compromise_probability < self._warning_threshold:
                return None

        severity = self._determine_severity(drift)
        if severity is None:
            return None

        return CompromiseAlert(
            agent_id=drift.agent_id,
            severity=severity,
            drift_category=drift.drift_category,
            compromise_probability=drift.compromise_probability,
            geodesic_distance=drift.geodesic_distance,
            drift_magnitude=drift.drift_magnitude,
            evidence={
                "geodesic_distance": drift.geodesic_distance,
                "euclidean_distance": drift.euclidean_distance,
                "cosine_similarity": drift.cosine_similarity,
                "drift_category": drift.drift_category.value,
                "per_dimension_drift": drift.per_dimension_drift,
                "p_value": drift.p_value,
                "is_significant": drift.is_significant,
            },
            recommendation=self._recommend_action(severity, drift),
        )

    def _determine_severity(self, drift: DriftMeasurement) -> Optional[str]:
        """Determine alert severity from drift measurements."""
        prob = drift.compromise_probability
        dist = drift.geodesic_distance

        if prob >= self._critical_threshold or dist >= self._distance_critical:
            return "critical"
        if prob >= self._warning_threshold or dist >= self._distance_warning:
            return "warning"
        return None

    def _recommend_action(self, severity: str, drift: DriftMeasurement) -> str:
        """Generate a recommended action based on severity and drift type."""
        if severity == "critical":
            return (
                f"CRITICAL: Agent {drift.agent_id} shows {drift.drift_category.value} drift "
                f"with {drift.compromise_probability:.0%} compromise probability. "
                f"Recommend immediate suspension and re-baseline."
            )
        return (
            f"WARNING: Agent {drift.agent_id} shows elevated {drift.drift_category.value} drift "
            f"(magnitude={drift.drift_magnitude:.2f}). "
            f"Monitor closely and consider re-validation."
        )

    def evaluate_multiple(
        self, drifts: list[DriftMeasurement]
    ) -> list[CompromiseAlert]:
        """Evaluate multiple drift measurements and return all alerts."""
        alerts = []
        for drift in drifts:
            alert = self.evaluate(drift)
            if alert is not None:
                alerts.append(alert)
        return alerts
