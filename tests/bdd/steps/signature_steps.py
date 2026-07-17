import sys
from pathlib import Path

import numpy as np
from behave import given, then, when

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import SignatureType
from domain.models import AgentProfile
from engine.geometric.distance import geodesic_distance
from engine.signature_generator import SignatureGenerator


@given('an agent "{name}" with model "{model}"')
def step_agent_with_model(context, name, model):
    context.agent = AgentProfile(
        agent_id=name, display_name=f"Agent {name}", model_id=model,
    )
    context.adapter = MockInferenceAdapter()
    context.extractor = DefaultMetricExtractor()
    context.generator = SignatureGenerator(manifold_method="pca")


@given('{n:d} controlled runs on the "{scenario}" scenario')
def step_controlled_runs(context, n, scenario):
    context.metrics_per_run = []
    context.run_ids = []
    for i in range(n):
        run = context.adapter.execute(context.agent, f"Prompt {i}")
        metrics = context.extractor.extract(run)
        context.metrics_per_run.append(metrics)
        context.run_ids.append(run.run_id)


@when("a baseline signature is computed")
def step_compute_baseline(context):
    context.signature = context.generator.generate(
        context.agent.agent_id, context.metrics_per_run, context.run_ids,
        signature_type=SignatureType.BASELINE,
    )


@then("the signature has a stability score above {threshold:g}")
def step_stability_above(context, threshold):
    assert context.signature.stability_score > threshold, (
        f"Stability {context.signature.stability_score} <= {threshold}"
    )


@then("the signature embedding dimension is {dim:d}")
def step_embedding_dim(context, dim):
    assert context.signature.embedding_dimension == dim


@then("the signature contains a metric tensor")
def step_has_tensor(context):
    assert context.signature.metric_tensor is not None


@given('an agent "{name}" with default behavior')
def step_agent_default(context, name):
    context.agents = getattr(context, "agents", {})
    context.agents[name] = {
        "agent": AgentProfile(agent_id=name, display_name=name, model_id="claude-sonnet-4-20250514"),
        "adapter": MockInferenceAdapter(),
    }


@given('an agent "{name}" with code-heavy behavior')
def step_agent_code(context, name):
    context.agents = getattr(context, "agents", {})
    context.agents[name] = {
        "agent": AgentProfile(agent_id=name, display_name=name, model_id="claude-opus-4-20250514"),
        "adapter": MockInferenceAdapter(
            response_key="code", latency_ms=300, input_tokens=200,
            output_tokens=120, thinking_tokens=80, include_tool_calls=True,
        ),
    }


@given("{n:d} controlled runs for each agent")
def step_runs_for_each(context, n):
    context.extractor = DefaultMetricExtractor()
    context.generator = SignatureGenerator(manifold_method="pca")
    for name, data in context.agents.items():
        metrics_list = []
        run_ids = []
        for i in range(n):
            run = data["adapter"].execute(data["agent"], f"Prompt {i}")
            metrics_list.append(context.extractor.extract(run))
            run_ids.append(run.run_id)
        data["metrics"] = metrics_list
        data["run_ids"] = run_ids


@when("baseline signatures are computed for both")
def step_baselines_both(context):
    context.signatures = {}
    for name, data in context.agents.items():
        sig = context.generator.generate(
            name, data["metrics"], data["run_ids"],
            signature_type=SignatureType.BASELINE,
        )
        context.signatures[name] = sig


@then("the geodesic distance between signatures is above {threshold:g}")
def step_distance_above(context, threshold):
    sigs = list(context.signatures.values())
    vec_a = np.array(sigs[0].embedding_vector)
    vec_b = np.array(sigs[1].embedding_vector)
    dist = geodesic_distance(vec_a, vec_b)
    assert dist > threshold, f"Distance {dist} <= {threshold}"
