from __future__ import annotations

import sys
from pathlib import Path

from behave import given, when, then

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from domain.enums import SignatureType
from domain.geometry import GeometricSignature
from domain.temporal import DriftPattern
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


@given("a baseline signature at the origin")
def step_baseline_origin(context):
    context.agent_id = "temporal-test-agent"
    context.baseline = _make_sig(context.agent_id, [0.0, 0.0, 0.0, 0.0])
    context.tracker = TemporalTracker(window_size=3)


@given("{n:d} snapshots with gradually increasing distances")
def step_gradual_snapshots(context, n):
    distances = [0.1 + 0.05 * i for i in range(n)]
    context.snapshots = [
        _make_sig(context.agent_id, [d, 0.0, 0.0, 0.0]) for d in distances
    ]


@given("{n:d} snapshots with a sudden jump at index {idx:d}")
def step_jump_snapshots(context, n, idx):
    distances = [0.1] * n
    distances[idx] = 5.0
    context.snapshots = [
        _make_sig(context.agent_id, [d, 0.0, 0.0, 0.0]) for d in distances
    ]


@given("{n:d} snapshots with constant distances")
def step_constant_snapshots(context, n):
    context.snapshots = [
        _make_sig(context.agent_id, [0.1, 0.0, 0.0, 0.0]) for _ in range(n)
    ]


@when("temporal drift is tracked")
def step_track(context):
    context.report = context.tracker.track(
        context.agent_id, context.snapshots, context.baseline
    )


@then('the pattern is "{pattern_name}"')
def step_pattern_is(context, pattern_name):
    expected = DriftPattern(pattern_name)
    assert context.report.pattern == expected, (
        f"Expected {expected}, got {context.report.pattern}"
    )


@then("drift velocity is positive")
def step_velocity_positive(context):
    assert context.report.drift_velocity > 0, (
        f"Expected positive velocity, got {context.report.drift_velocity}"
    )


@then("index {idx:d} is flagged as an anomaly")
def step_anomaly_index(context, idx):
    assert idx in context.report.anomaly_indices, (
        f"Index {idx} not in anomalies: {context.report.anomaly_indices}"
    )


@then("no anomalies are detected")
def step_no_anomalies(context):
    assert len(context.report.anomaly_indices) == 0, (
        f"Expected no anomalies, got {context.report.anomaly_indices}"
    )
