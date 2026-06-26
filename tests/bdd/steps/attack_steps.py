from __future__ import annotations

import sys
from pathlib import Path

from behave import given, when, then

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import RealisticMockAdapter
from domain.attacks import AttackResult, AttackType
from domain.enums import SignatureType
from domain.geometry import GeometricSignature
from domain.models import AgentProfile
from engine.attack_simulator import AttackSimulator
from engine.canary_system import CanarySystem
from engine.drift_detector import DriftDetector
from engine.semantic_analyzer import SemanticAnalyzer
from engine.signature_generator import SignatureGenerator
from engine.temporal_tracker import TemporalTracker


@given("an attack simulator with all detection engines")
def step_create_simulator(context):
    extractor = DefaultMetricExtractor()
    generator = SignatureGenerator(min_runs=5)
    drift_detector = DriftDetector()
    adapter = RealisticMockAdapter(profile="coder")
    semantic_analyzer = SemanticAnalyzer(adapter=adapter)
    canary_system = CanarySystem()
    temporal_tracker = TemporalTracker(window_size=3)

    context.simulator = AttackSimulator(
        extractor=extractor,
        generator=generator,
        drift_detector=drift_detector,
        semantic_analyzer=semantic_analyzer,
        canary_system=canary_system,
        temporal_tracker=temporal_tracker,
    )
    context.adapter = adapter


@given("a target agent with a baseline signature")
def step_create_agent_and_baseline(context):
    context.agent = AgentProfile(
        agent_id="bdd-attack-agent",
        display_name="BDD Attack Agent",
        model_id="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant.",
    )
    context.target_baseline = GeometricSignature(
        agent_id="bdd-attack-agent",
        signature_type=SignatureType.BASELINE,
        embedding_vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        embedding_dimension=7,
        manifold_coordinates=[0.5, 0.3],
        metric_snapshot={"avg_response_length": 0.5},
        run_ids=["r1", "r2"],
        num_runs=2,
        computation_method="test",
        stability_score=0.85,
    )


@when("all 4 attacks are simulated")
def step_run_all_attacks(context):
    context.results = context.simulator.run_all_attacks(
        context.target_baseline, context.adapter, context.agent,
    )


@when("a summary report is generated")
def step_generate_summary(context):
    context.report = context.simulator.summary_report(context.results)


@then("each attack produces an AttackResult")
def step_check_attack_results(context):
    assert len(context.results) == 4
    for result in context.results:
        assert isinstance(result, AttackResult)


@then("each result has a valid attack type")
def step_check_attack_types(context):
    valid_types = set(AttackType)
    for result in context.results:
        assert result.attack_type in valid_types


@then("each result has detection_rate in [0, 1]")
def step_check_detection_rate(context):
    for result in context.results:
        assert 0.0 <= result.detection_rate <= 1.0, (
            f"detection_rate={result.detection_rate} out of range for "
            f"{result.attack_type}"
        )


@then("each result has evasion_rate = 1 - detection_rate")
def step_check_evasion_rate(context):
    for result in context.results:
        assert abs(result.detection_rate + result.evasion_rate - 1.0) < 1e-9, (
            f"evasion_rate={result.evasion_rate} != 1 - detection_rate="
            f"{result.detection_rate} for {result.attack_type}"
        )


@then("the report contains all 4 attack types")
def step_check_report_types(context):
    for atype in AttackType:
        assert atype.value in context.report["per_attack"], (
            f"Missing attack type {atype.value} in report"
        )


@then("the report has overall detection and evasion rates")
def step_check_report_rates(context):
    assert "overall_detection_rate" in context.report
    assert "overall_evasion_rate" in context.report
    assert 0.0 <= context.report["overall_detection_rate"] <= 1.0
    assert 0.0 <= context.report["overall_evasion_rate"] <= 1.0
