from __future__ import annotations

from domain.enums import SignatureType
from domain.geometry import GeometricSignature
from domain.temporal import DriftPattern, TemporalDriftReport, TemporalWindow
from engine.temporal_tracker import TemporalTracker


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


def _build_report() -> TemporalDriftReport:
    distances = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    baseline = _make_sig("contract-agent", [0.0, 0.0, 0.0, 0.0])
    sigs = [_make_sig("contract-agent", [d, 0.0, 0.0, 0.0]) for d in distances]
    tracker = TemporalTracker(window_size=3)
    return tracker.track("contract-agent", sigs, baseline)


class TestTemporalContracts:
    def test_output_is_temporal_drift_report(self):
        report = _build_report()
        assert isinstance(report, TemporalDriftReport)

    def test_windows_contain_temporal_window_instances(self):
        report = _build_report()
        assert len(report.windows) > 0
        for w in report.windows:
            assert isinstance(w, TemporalWindow)

    def test_pattern_is_valid_drift_pattern(self):
        report = _build_report()
        assert isinstance(report.pattern, DriftPattern)
        assert report.pattern.value in [p.value for p in DriftPattern]
