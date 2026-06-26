from __future__ import annotations

import sys
from pathlib import Path

from behave import given, then, when

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import DriftCategory, SignatureType
from domain.geometry import DriftMeasurement
from domain.models import AgentProfile
from engine.secure_measurement import SecureMeasurement
from engine.signature_generator import SignatureGenerator


def _make_sig(agent_id, n=5):
    adapter = MockInferenceAdapter()
    extractor = DefaultMetricExtractor()
    generator = SignatureGenerator(manifold_method="pca")
    agent = AgentProfile(agent_id=agent_id, display_name=agent_id, model_id="test")
    metrics_per_run = []
    run_ids = []
    for i in range(n):
        run = adapter.execute(agent, f"Prompt {i}")
        metrics_per_run.append(extractor.extract(run))
        run_ids.append(run.run_id)
    return generator.generate(agent_id, metrics_per_run, run_ids, SignatureType.BASELINE)


# ---- Scenario 1: Signature encryption protects vector at rest ----


@given('an agent "{name}" with a geometric signature')
def step_agent_with_signature(context, name):
    context.sm = SecureMeasurement(encryption_key="bdd-test-key")
    context.signature = _make_sig(name)


@when("the signature is encrypted")
def step_encrypt_signature(context):
    context.envelope = context.sm.encrypt_signature(context.signature)


@then("the encrypted envelope differs from the raw vector")
def step_encrypted_differs(context):
    import json

    raw_json = json.dumps(context.signature.embedding_vector)
    assert context.envelope.encrypted_vector != raw_json


@then("decrypting the envelope recovers the original vector")
def step_decrypt_recovers(context):
    recovered = context.sm.decrypt_signature(context.envelope)
    assert recovered == context.signature.embedding_vector


# ---- Scenario 2: Commitment hash detects tampering ----


@when("the commitment hash is verified against the original vector")
def step_verify_original(context):
    context.verify_result = context.sm.verify_commitment(
        context.envelope, context.signature.embedding_vector
    )


@then("the verification succeeds")
def step_verification_succeeds(context):
    assert context.verify_result is True


@then("verification against a tampered vector fails")
def step_verification_tampered_fails(context):
    tampered = [v + 1.0 for v in context.signature.embedding_vector]
    result = context.sm.verify_commitment(context.envelope, tampered)
    assert result is False


# ---- Scenario 3: Drift obfuscation hides exact dimensions ----


@given("a drift measurement with known per-dimension values")
def step_known_drift(context):
    context.sm = SecureMeasurement(encryption_key="bdd-test-key-drift")
    context.drift = DriftMeasurement(
        agent_id="test-bdd",
        baseline_signature_id="s1",
        current_signature_id="s2",
        geodesic_distance=0.5,
        euclidean_distance=0.4,
        cosine_similarity=0.85,
        drift_category=DriftCategory.SEMANTIC,
        drift_magnitude=0.3,
        per_dimension_drift={
            "response_structure": 0.12,
            "token_economics": 0.08,
            "reasoning_pattern": 0.15,
        },
        is_significant=True,
        p_value=0.03,
        compromise_probability=0.4,
    )


@when("the drift is obfuscated with noise")
def step_obfuscate(context):
    context.obfuscated = context.sm.obfuscate_drift(context.drift, noise_scale=0.5)


@then("the obfuscated dimensions differ from the originals")
def step_dims_differ(context):
    any_different = any(
        context.obfuscated.obfuscated_dimensions[k] != context.drift.per_dimension_drift[k]
        for k in context.drift.per_dimension_drift
    )
    assert any_different


@then("the severity classification is preserved")
def step_severity_preserved(context):
    assert context.obfuscated.severity == "warning"
