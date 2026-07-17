from __future__ import annotations

from domain.enums import SignatureType
from domain.geometry import GeometricSignature
from domain.temporal import DriftPattern, TemporalDriftReport
from engine.temporal_tracker import TemporalTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sig(agent_id: str, vector: list[float]) -> GeometricSignature:
    return GeometricSignature(
        agent_id=agent_id,
        signature_type=SignatureType.SNAPSHOT,
        embedding_vector=vector,
        embedding_dimension=len(vector),
        manifold_coordinates=[0.0, 0.0],
        metric_snapshot={},
        run_ids=["r1"],
        num_runs=1,
        computation_method="test",
    )


def _sigs_from_distances(
    agent_id: str, distances: list[float], dim: int = 4
) -> tuple[GeometricSignature, list[GeometricSignature]]:
    """Create a baseline at the origin and snapshots whose Euclidean
    distance from that baseline matches the requested *distances*.

    Each snapshot places its entire offset along the first axis so that
    ``||snapshot - baseline|| == distance``.
    """
    baseline = _make_sig(agent_id, [0.0] * dim)
    sigs: list[GeometricSignature] = []
    for d in distances:
        vec = [d] + [0.0] * (dim - 1)
        sigs.append(_make_sig(agent_id, vec))
    return baseline, sigs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTemporalTracker:
    def test_track_returns_temporal_drift_report(self):
        distances = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        baseline, sigs = _sigs_from_distances("agent-a", distances)
        tracker = TemporalTracker(window_size=3)
        report = tracker.track("agent-a", sigs, baseline)
        assert isinstance(report, TemporalDriftReport)
        assert report.agent_id == "agent-a"

    def test_sliding_windows_cover_full_range(self):
        distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        tracker = TemporalTracker(window_size=3, step_size=1)
        windows = tracker.compute_sliding_windows("a", distances)
        # First window starts at 0, last window ends at len-1
        assert windows[0].window_start == 0
        assert windows[-1].window_end == len(distances) - 1

    def test_window_size_respected(self):
        distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        tracker = TemporalTracker(window_size=4, step_size=1)
        windows = tracker.compute_sliding_windows("a", distances)
        for w in windows:
            assert w.window_size == 4

    def test_classify_stable(self):
        distances = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        tracker = TemporalTracker(window_size=3)
        pattern, confidence = tracker.classify_pattern(
            distances, tracker.compute_sliding_windows("a", distances)
        )
        assert pattern == DriftPattern.STABLE
        assert confidence > 0.0

    def test_classify_gradual_accumulation(self):
        distances = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
        tracker = TemporalTracker(window_size=3)
        pattern, confidence = tracker.classify_pattern(
            distances, tracker.compute_sliding_windows("a", distances)
        )
        assert pattern == DriftPattern.GRADUAL_ACCUMULATION
        assert confidence > 0.0

    def test_classify_sudden_jump(self):
        distances = [0.1, 0.1, 0.1, 5.0, 0.1, 0.1, 0.1, 0.1]
        tracker = TemporalTracker(window_size=3)
        pattern, confidence = tracker.classify_pattern(
            distances, tracker.compute_sliding_windows("a", distances)
        )
        assert pattern == DriftPattern.SUDDEN_JUMP
        assert confidence > 0.0

    def test_classify_oscillation(self):
        distances = [0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5]
        tracker = TemporalTracker(window_size=3)
        pattern, confidence = tracker.classify_pattern(
            distances, tracker.compute_sliding_windows("a", distances)
        )
        assert pattern == DriftPattern.OSCILLATION
        assert confidence > 0.0

    def test_classify_mean_reversion(self):
        distances = [0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.05]
        tracker = TemporalTracker(window_size=3)
        pattern, confidence = tracker.classify_pattern(
            distances, tracker.compute_sliding_windows("a", distances)
        )
        assert pattern == DriftPattern.MEAN_REVERSION
        assert confidence > 0.0

    def test_classify_permanent_shift(self):
        distances = [0.1, 0.3, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7]
        tracker = TemporalTracker(window_size=3)
        pattern, confidence = tracker.classify_pattern(
            distances, tracker.compute_sliding_windows("a", distances)
        )
        assert pattern == DriftPattern.PERMANENT_SHIFT
        assert confidence > 0.0

    def test_detect_anomalies_flags_outliers(self):
        distances = [0.1, 0.1, 0.1, 5.0, 0.1, 0.1, 0.1, 0.1]
        tracker = TemporalTracker()
        anomalies = tracker.detect_anomalies(distances)
        assert 3 in anomalies
        assert len(anomalies) == 1

    def test_drift_velocity_positive_for_increasing(self):
        distances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        baseline, sigs = _sigs_from_distances("agent-a", distances)
        tracker = TemporalTracker(window_size=3)
        report = tracker.track("agent-a", sigs, baseline)
        assert report.drift_velocity > 0

    def test_drift_acceleration_detects_rate_change(self):
        # Quadratic growth: acceleration should be positive
        distances = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]
        baseline, sigs = _sigs_from_distances("agent-a", distances)
        tracker = TemporalTracker(window_size=3)
        report = tracker.track("agent-a", sigs, baseline)
        assert report.drift_acceleration > 0
