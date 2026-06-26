import sys
from pathlib import Path

from behave import given, when, then

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from domain.enums import DriftCategory
from domain.geometry import DriftMeasurement
from engine.compromise_detector import CompromiseDetector


@given("a drift measurement with high compromise probability")
def step_high_compromise(context):
    context.compromise_detector = CompromiseDetector()
    context.drift_measurement = DriftMeasurement(
        agent_id=context.agent_name,
        baseline_signature_id="sig-baseline",
        current_signature_id="sig-current",
        geodesic_distance=1.5,
        euclidean_distance=1.2,
        cosine_similarity=0.4,
        drift_category=DriftCategory.GOAL,
        drift_magnitude=0.9,
        per_dimension_drift={"response_structure": 0.8},
        is_significant=True,
        p_value=0.001,
        compromise_probability=0.95,
    )


@given("a drift measurement with low compromise probability")
def step_low_compromise(context):
    context.compromise_detector = CompromiseDetector()
    context.drift_measurement = DriftMeasurement(
        agent_id=context.agent_name,
        baseline_signature_id="sig-baseline",
        current_signature_id="sig-current",
        geodesic_distance=0.05,
        euclidean_distance=0.04,
        cosine_similarity=0.99,
        drift_category=DriftCategory.SEMANTIC,
        drift_magnitude=0.05,
        per_dimension_drift={"response_structure": 0.02},
        is_significant=False,
        p_value=0.8,
        compromise_probability=0.02,
    )


@when("the compromise detector evaluates the drift")
def step_evaluate_compromise(context):
    context.alert = context.compromise_detector.evaluate(context.drift_measurement)


@then("an alert is generated")
def step_alert_generated(context):
    assert context.alert is not None


@then('the alert severity is "warning" or "critical"')
def step_alert_severity(context):
    assert context.alert.severity in ("warning", "critical")


@then("no alert is generated")
def step_no_alert(context):
    assert context.alert is None
