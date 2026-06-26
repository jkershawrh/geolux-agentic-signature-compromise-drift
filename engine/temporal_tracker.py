from __future__ import annotations

import numpy as np

from domain.geometry import GeometricSignature
from domain.temporal import DriftPattern, TemporalDriftReport, TemporalWindow
from engine.geometric.distance import euclidean_distance


class TemporalTracker:
    """Sliding-window analysis to detect temporal drift patterns.

    Tracks how an agent's behavioural signature evolves over a sequence of
    snapshots relative to a baseline, classifying the overall pattern as
    stable, gradual accumulation, sudden jump, oscillation, mean-reversion,
    or permanent shift.
    """

    def __init__(
        self,
        window_size: int = 5,
        step_size: int = 1,
        anomaly_threshold_sigma: float = 2.0,
    ):
        self._window_size = window_size
        self._step_size = step_size
        self._anomaly_threshold_sigma = anomaly_threshold_sigma

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track(
        self,
        agent_id: str,
        signatures: list[GeometricSignature],
        baseline: GeometricSignature,
    ) -> TemporalDriftReport:
        """Analyse a time-ordered sequence of signatures against a baseline.

        Returns a *TemporalDriftReport* summarising the drift trajectory.
        """
        baseline_vec = np.array(baseline.embedding_vector)
        distances = [
            euclidean_distance(baseline_vec, np.array(sig.embedding_vector))
            for sig in signatures
        ]

        windows = self.compute_sliding_windows(agent_id, distances)
        pattern, confidence = self.classify_pattern(distances, windows)
        anomalies = self.detect_anomalies(distances)
        velocity = self._compute_velocity(distances)
        acceleration = self._compute_acceleration(distances)
        cumulative = float(sum(distances))

        return TemporalDriftReport(
            agent_id=agent_id,
            windows=windows,
            pattern=pattern,
            pattern_confidence=confidence,
            cumulative_drift=cumulative,
            drift_velocity=velocity,
            drift_acceleration=acceleration,
            anomaly_indices=anomalies,
        )

    # ------------------------------------------------------------------
    # Sliding windows
    # ------------------------------------------------------------------

    def compute_sliding_windows(
        self, agent_id: str, distances: list[float]
    ) -> list[TemporalWindow]:
        """Build sliding windows over the distance array."""
        windows: list[TemporalWindow] = []
        n = len(distances)
        start = 0
        while start + self._window_size <= n:
            end = start + self._window_size
            window_distances = distances[start:end]
            trend = self._compute_trend(window_distances)
            windows.append(
                TemporalWindow(
                    agent_id=agent_id,
                    window_start=start,
                    window_end=end - 1,
                    window_size=self._window_size,
                    mean_distance=float(np.mean(window_distances)),
                    max_distance=float(np.max(window_distances)),
                    distance_trend=trend,
                )
            )
            start += self._step_size
        return windows

    # ------------------------------------------------------------------
    # Pattern classification
    # ------------------------------------------------------------------

    def classify_pattern(
        self,
        distances: list[float],
        windows: list[TemporalWindow],
    ) -> tuple[DriftPattern, float]:
        """Classify the temporal drift pattern and return a confidence score."""
        arr = np.array(distances, dtype=float)
        n = len(arr)
        if n < 2:
            return DriftPattern.STABLE, 1.0

        mean_d = float(np.mean(arr))
        std_d = float(np.std(arr))
        overall_trend = self._compute_trend(distances)
        diffs = np.diff(arr)

        # Build scores for each candidate pattern.
        scores: dict[DriftPattern, float] = {}

        # SUDDEN_JUMP — any outlier dominates.
        # When a jump is present the data statistics (trend, variance) are
        # distorted, so we give SUDDEN_JUMP priority and suppress patterns
        # that would be artifacts of the spike.
        threshold = mean_d + self._anomaly_threshold_sigma * std_d
        has_jump = bool(np.any(arr > threshold)) and std_d > 0
        if has_jump:
            max_excess = float(np.max(arr) - mean_d) / (std_d if std_d > 0 else 1.0)
            scores[DriftPattern.SUDDEN_JUMP] = min(1.0, max_excess / 5.0 + 0.5)
        else:
            scores[DriftPattern.SUDDEN_JUMP] = 0.0

        # STABLE — low variance, no significant trend
        rel_std = std_d / mean_d if mean_d > 0 else std_d
        stable_score = 0.0
        if rel_std < 0.1 and abs(overall_trend) < 0.01:
            stable_score = 1.0 - rel_std * 10  # closer to 0 variance = higher score
        scores[DriftPattern.STABLE] = max(0.0, stable_score)

        # OSCILLATION — frequent sign changes in first differences.
        # Check this before directional patterns because oscillation
        # can produce spurious trend signals.
        if len(diffs) > 1:
            sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
            oscillation_ratio = sign_changes / len(diffs)
            if oscillation_ratio > 0.6:
                scores[DriftPattern.OSCILLATION] = min(1.0, oscillation_ratio + 0.2)
            else:
                scores[DriftPattern.OSCILLATION] = 0.0
        else:
            scores[DriftPattern.OSCILLATION] = 0.0

        # MEAN_REVERSION — positive trend first half, meaningfully negative
        # second half (not just floating-point noise around zero).
        half = n // 2
        if half >= 2 and n - half >= 2:
            trend_first = self._compute_trend(distances[:half])
            trend_second = self._compute_trend(distances[half:])
            if trend_first > 0.005 and trend_second < -0.005:
                scores[DriftPattern.MEAN_REVERSION] = min(
                    1.0, (trend_first - trend_second) * 5 + 0.3
                )
            else:
                scores[DriftPattern.MEAN_REVERSION] = 0.0
        else:
            scores[DriftPattern.MEAN_REVERSION] = 0.0

        # PERMANENT_SHIFT — increases in first 50 %, then plateaus in last 30 %.
        # Only when there is no jump and not already oscillation.
        cutoff_70 = max(1, int(n * 0.7))
        if half >= 2 and n - cutoff_70 >= 2 and not has_jump:
            trend_first_half = self._compute_trend(distances[:half])
            trend_last_30 = self._compute_trend(distances[cutoff_70:])
            if trend_first_half > 0.01 and abs(trend_last_30) < 0.01:
                scores[DriftPattern.PERMANENT_SHIFT] = min(
                    1.0, trend_first_half * 10 + 0.4
                )
            else:
                scores[DriftPattern.PERMANENT_SHIFT] = 0.0
        else:
            scores[DriftPattern.PERMANENT_SHIFT] = 0.0

        # GRADUAL_ACCUMULATION — positive trend, end > start, no jumps,
        # and not a permanent shift (which also trends up then flattens).
        if (
            overall_trend > 0.01
            and arr[-1] > arr[0]
            and not has_jump
            and scores.get(DriftPattern.PERMANENT_SHIFT, 0) == 0
        ):
            scores[DriftPattern.GRADUAL_ACCUMULATION] = min(
                1.0, overall_trend * 10 + 0.3
            )
        else:
            scores[DriftPattern.GRADUAL_ACCUMULATION] = 0.0

        # Pick best pattern
        sorted_patterns = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_pattern, best_score = sorted_patterns[0]
        second_score = sorted_patterns[1][1] if len(sorted_patterns) > 1 else 0.0

        # If nothing scored, default to STABLE
        if best_score == 0.0:
            return DriftPattern.STABLE, 1.0

        # Confidence = how much the best score separates from the runner-up
        if best_score > 0:
            confidence = 1.0 - (second_score / best_score)
        else:
            confidence = 1.0
        confidence = max(0.01, min(1.0, confidence))

        return best_pattern, confidence

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def detect_anomalies(self, distances: list[float]) -> list[int]:
        """Return indices where the distance exceeds mean + threshold * std."""
        arr = np.array(distances, dtype=float)
        mean_d = float(np.mean(arr))
        std_d = float(np.std(arr))
        threshold = mean_d + self._anomaly_threshold_sigma * std_d
        return [int(i) for i, d in enumerate(arr) if d > threshold]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_trend(values: list[float]) -> float:
        """Linear regression slope over the value sequence."""
        n = len(values)
        if n < 2:
            return 0.0
        coeffs = np.polyfit(range(n), values, 1)
        return float(coeffs[0])

    @staticmethod
    def _compute_velocity(distances: list[float]) -> float:
        """Mean of first differences (rate of distance change)."""
        if len(distances) < 2:
            return 0.0
        diffs = np.diff(distances)
        return float(np.mean(diffs))

    @staticmethod
    def _compute_acceleration(distances: list[float]) -> float:
        """Mean of second differences (rate of velocity change)."""
        if len(distances) < 3:
            return 0.0
        second_diffs = np.diff(distances, n=2)
        return float(np.mean(second_diffs))
