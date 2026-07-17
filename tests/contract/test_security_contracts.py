"""Contract tests verifying secure measurement components satisfy expected interfaces."""
from __future__ import annotations

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import DriftCategory, SignatureType
from domain.geometry import DriftMeasurement
from domain.models import AgentProfile
from domain.security import ObfuscatedDriftResult, SecureSignatureEnvelope
from engine.drift_detector import DriftDetector
from engine.secure_measurement import SecureMeasurement
from engine.signature_generator import SignatureGenerator


def _make_sig(agent_id, adapter_kwargs=None, n=5):
    kwargs = adapter_kwargs or {}
    adapter = MockInferenceAdapter(**kwargs)
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


def _make_drift():
    return DriftMeasurement(
        agent_id="contract-agent",
        baseline_signature_id="s1",
        current_signature_id="s2",
        geodesic_distance=0.5,
        euclidean_distance=0.4,
        cosine_similarity=0.85,
        drift_category=DriftCategory.SEMANTIC,
        drift_magnitude=0.3,
        per_dimension_drift={"response_structure": 0.12},
        is_significant=True,
        p_value=0.03,
        compromise_probability=0.4,
    )


class TestSecureMeasurementContracts:
    def test_encrypt_signature_returns_envelope(self):
        """encrypt_signature must return a SecureSignatureEnvelope."""
        sm = SecureMeasurement(encryption_key="contract-key")
        sig = _make_sig("contract-agent")
        result = sm.encrypt_signature(sig)
        assert isinstance(result, SecureSignatureEnvelope)
        assert result.agent_id == sig.agent_id
        assert result.signature_id == sig.signature_id
        assert isinstance(result.encrypted_vector, str)
        assert isinstance(result.commitment_hash, str)

    def test_obfuscate_drift_returns_obfuscated_result(self):
        """obfuscate_drift must return an ObfuscatedDriftResult."""
        sm = SecureMeasurement(encryption_key="contract-key")
        drift = _make_drift()
        result = sm.obfuscate_drift(drift)
        assert isinstance(result, ObfuscatedDriftResult)
        assert result.agent_id == drift.agent_id
        assert isinstance(result.severity, str)
        assert result.severity in ("none", "warning", "critical")
        assert isinstance(result.obfuscated_dimensions, dict)

    def test_secure_compare_returns_obfuscated_result(self):
        """secure_compare must return an ObfuscatedDriftResult."""
        sm = SecureMeasurement(encryption_key="contract-key")
        sig = _make_sig("contract-agent")
        envelope = sm.encrypt_signature(sig)
        current = _make_sig("contract-agent", {"response_key": "code", "latency_ms": 500})
        detector = DriftDetector(n_permutations=10)
        result = sm.secure_compare(envelope, current, detector)
        assert isinstance(result, ObfuscatedDriftResult)
        assert result.raw_distance_used is False
