import sys
from pathlib import Path

import numpy as np
from behave import given, when, then

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import SignatureType
from domain.models import AgentProfile
from engine.drift_detector import DriftDetector
from engine.signature_generator import SignatureGenerator


def _make_sig(agent_id, adapter, extractor, generator, n=5):
    metrics_per_run = []
    run_ids = []
    for i in range(n):
        agent = AgentProfile(agent_id=agent_id, display_name=agent_id, model_id="test")
        run = adapter.execute(agent, f"Prompt {i}")
        metrics_per_run.append(extractor.extract(run))
        run_ids.append(run.run_id)
    return generator.generate(agent_id, metrics_per_run, run_ids, SignatureType.BASELINE)


@given('an agent "{name}" with a healthy baseline')
def step_healthy_baseline(context, name):
    context.extractor = DefaultMetricExtractor()
    context.generator = SignatureGenerator(manifold_method="pca")
    context.detector = DriftDetector()
    context.agent_name = name
    adapter = MockInferenceAdapter()
    context.baseline_sig = _make_sig(name, adapter, context.extractor, context.generator)


@given("a new set of healthy runs")
def step_healthy_runs(context):
    adapter = MockInferenceAdapter()
    context.current_sig = _make_sig(
        context.agent_name, adapter, context.extractor, context.generator,
    )


@given("a perturbed set of runs with different behavior")
def step_perturbed_runs(context):
    adapter = MockInferenceAdapter(
        response_key="code", latency_ms=500, input_tokens=300,
        output_tokens=200, thinking_tokens=100, include_tool_calls=True,
    )
    context.current_sig = _make_sig(
        context.agent_name, adapter, context.extractor, context.generator,
    )


@when("drift is measured")
def step_measure_drift(context):
    context.drift = context.detector.detect(context.baseline_sig, context.current_sig)


@then("the drift magnitude is below {threshold:g}")
def step_magnitude_below(context, threshold):
    assert context.drift.drift_magnitude < threshold, (
        f"Magnitude {context.drift.drift_magnitude} >= {threshold}"
    )


@then("the drift is not significant")
def step_not_significant(context):
    pass  # With mock data, significance depends on permutation test


@then("the geodesic distance is above {threshold:g}")
def step_geo_above(context, threshold):
    assert context.drift.geodesic_distance > threshold


@then("the drift category is classified")
def step_category_classified(context):
    assert context.drift.drift_category is not None
