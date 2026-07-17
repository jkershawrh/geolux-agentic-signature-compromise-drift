import sys
from pathlib import Path

from behave import given, then, when

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import SignatureType
from domain.models import AgentProfile
from engine.authentication import AuthenticationEngine
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


@given('an agent "{name}" with an established baseline')
def step_established_baseline(context, name):
    context.extractor = DefaultMetricExtractor()
    context.generator = SignatureGenerator(manifold_method="pca")
    context.auth_engine = AuthenticationEngine()
    context.agent_name = name
    adapter = MockInferenceAdapter()
    context.baseline = _make_sig(name, adapter, context.extractor, context.generator)


@given("a new signature from the same agent")
def step_same_agent_sig(context):
    adapter = MockInferenceAdapter()
    context.current = _make_sig(
        context.agent_name, adapter, context.extractor, context.generator,
    )


@given('a signature from a different agent "{name}"')
def step_different_agent_sig(context, name):
    adapter = MockInferenceAdapter(
        response_key="code", latency_ms=500, input_tokens=300,
        output_tokens=200, thinking_tokens=100, include_tool_calls=True,
    )
    context.current = _make_sig(name, adapter, context.extractor, context.generator)


@when("authentication is performed")
def step_authenticate(context):
    context.auth_result = context.auth_engine.verify(context.current, context.baseline)


@when("authentication is performed against alpha's baseline")
def step_authenticate_against_alpha(context):
    context.auth_result = context.auth_engine.verify(context.current, context.baseline)


@then("the agent is verified as authentic")
def step_is_authentic(context):
    assert context.auth_result.is_authentic is True


@then("the confidence score is above {threshold:g}")
def step_confidence_above(context, threshold):
    assert context.auth_result.confidence > threshold


@then("the agent is not verified as authentic")
def step_not_authentic(context):
    assert context.auth_result.is_authentic is False or context.auth_result.confidence < 0.9
