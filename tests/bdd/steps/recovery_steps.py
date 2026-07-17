import sys
from pathlib import Path

from behave import given, then, when

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import SignatureType
from domain.models import AgentProfile
from engine.recovery_engine import RecoveryEngine
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


@given('an agent "{name}" that was flagged as compromised')
def step_compromised_agent(context, name):
    context.agent = AgentProfile(
        agent_id=name, display_name=name, model_id="test",
    )
    context.adapter = MockInferenceAdapter()
    context.extractor = DefaultMetricExtractor()
    context.generator = SignatureGenerator(manifold_method="pca")
    context.old_baseline = _make_sig(
        name, context.adapter, context.extractor, context.generator,
    )


@given("a set of clean recovery prompts")
def step_recovery_prompts(context):
    context.recovery_prompts = [f"Recovery prompt {i}" for i in range(10)]


@when("recovery is attempted")
def step_attempt_recovery(context):
    recovery = RecoveryEngine(
        adapter=context.adapter,
        extractor=context.extractor,
        generator=context.generator,
        recovery_distance_threshold=1.0,
    )
    context.recovery_result = recovery.recover(
        context.agent, context.old_baseline, context.recovery_prompts,
    )


@then("the recovery succeeds")
def step_recovery_success(context):
    assert context.recovery_result.success is True


@then("the new baseline is close to the old baseline")
def step_baseline_close(context):
    assert context.recovery_result.distance_from_old < 1.0
