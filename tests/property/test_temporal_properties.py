from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from domain.enums import SignatureType
from domain.geometry import GeometricSignature
from domain.temporal import DriftPattern, TemporalDriftReport
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


class TestTemporalProperties:
    @given(
        distances=st.lists(
            st.floats(min_value=0.0, max_value=10.0),
            min_size=5,
            max_size=20,
        ),
    )
    @settings(max_examples=50)
    def test_pattern_confidence_in_range(self, distances):
        baseline = _make_sig("prop-agent", [0.0, 0.0, 0.0, 0.0])
        sigs = [
            _make_sig("prop-agent", [d, 0.0, 0.0, 0.0]) for d in distances
        ]
        tracker = TemporalTracker(window_size=3)
        report = tracker.track("prop-agent", sigs, baseline)
        assert 0.0 <= report.pattern_confidence <= 1.0

    @given(
        distances=st.lists(
            st.floats(min_value=0.0, max_value=10.0),
            min_size=5,
            max_size=20,
        ),
    )
    @settings(max_examples=50)
    def test_cumulative_drift_non_negative(self, distances):
        baseline = _make_sig("prop-agent", [0.0, 0.0, 0.0, 0.0])
        sigs = [
            _make_sig("prop-agent", [d, 0.0, 0.0, 0.0]) for d in distances
        ]
        tracker = TemporalTracker(window_size=3)
        report = tracker.track("prop-agent", sigs, baseline)
        assert report.cumulative_drift >= 0.0

    @given(
        distances=st.lists(
            st.floats(min_value=0.0, max_value=10.0),
            min_size=5,
            max_size=20,
        ),
    )
    @settings(max_examples=50)
    def test_anomaly_indices_within_bounds(self, distances):
        baseline = _make_sig("prop-agent", [0.0, 0.0, 0.0, 0.0])
        sigs = [
            _make_sig("prop-agent", [d, 0.0, 0.0, 0.0]) for d in distances
        ]
        tracker = TemporalTracker(window_size=3)
        report = tracker.track("prop-agent", sigs, baseline)
        for idx in report.anomaly_indices:
            assert 0 <= idx < len(distances)
