from __future__ import annotations

import numpy as np
import pytest

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import DriftCategory, SignatureType
from domain.geometry import DriftMeasurement
from domain.models import AgentProfile
from domain.security import ObfuscatedDriftResult
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


def _make_drift(agent_id="test", geodesic=0.5):
    return DriftMeasurement(
        agent_id=agent_id,
        baseline_signature_id="s1",
        current_signature_id="s2",
        geodesic_distance=geodesic,
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


class TestSecureMeasurement:
    def test_encrypt_decrypt_roundtrip(self):
        """Encrypt then decrypt must return the original vector."""
        sm = SecureMeasurement(encryption_key="test-key-roundtrip")
        sig = _make_sig("test")
        envelope = sm.encrypt_signature(sig)
        recovered = sm.decrypt_signature(envelope)
        assert recovered == sig.embedding_vector

    def test_commitment_hash_deterministic(self):
        """Same vector must produce the same commitment hash every time."""
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        h1 = SecureMeasurement.compute_commitment_hash(vector)
        h2 = SecureMeasurement.compute_commitment_hash(vector)
        assert h1 == h2

    def test_verify_commitment_valid(self):
        """Correct raw vector passes commitment verification."""
        sm = SecureMeasurement(encryption_key="test-key-commit")
        sig = _make_sig("test")
        envelope = sm.encrypt_signature(sig)
        assert sm.verify_commitment(envelope, sig.embedding_vector) is True

    def test_verify_commitment_tampered(self):
        """Tampered vector fails commitment verification."""
        sm = SecureMeasurement(encryption_key="test-key-tamper")
        sig = _make_sig("test")
        envelope = sm.encrypt_signature(sig)
        tampered = [v + 1.0 for v in sig.embedding_vector]
        assert sm.verify_commitment(envelope, tampered) is False

    def test_obfuscate_drift_adds_noise(self):
        """Obfuscated dimension values must differ from the originals."""
        sm = SecureMeasurement(encryption_key="test-key-noise")
        drift = _make_drift()
        result = sm.obfuscate_drift(drift, noise_scale=0.5)
        # With scale=0.5, at least one dimension should differ
        any_different = any(
            result.obfuscated_dimensions[k] != drift.per_dimension_drift[k]
            for k in drift.per_dimension_drift
        )
        assert any_different

    def test_obfuscate_drift_preserves_severity(self):
        """Severity classification is based on geodesic distance, not noise."""
        sm = SecureMeasurement(encryption_key="test-key-sev")
        drift_warning = _make_drift(geodesic=0.5)
        result = sm.obfuscate_drift(drift_warning)
        assert result.severity == "warning"

        drift_critical = _make_drift(geodesic=1.5)
        result_critical = sm.obfuscate_drift(drift_critical)
        assert result_critical.severity == "critical"

        drift_none = _make_drift(geodesic=0.1)
        result_none = sm.obfuscate_drift(drift_none)
        assert result_none.severity == "none"

    def test_obfuscated_result_hides_raw_distance(self):
        """raw_distance_used must always be False in obfuscated output."""
        sm = SecureMeasurement(encryption_key="test-key-hide")
        drift = _make_drift()
        result = sm.obfuscate_drift(drift)
        assert result.raw_distance_used is False

    def test_secure_compare_returns_obfuscated(self):
        """secure_compare must return ObfuscatedDriftResult with no raw vectors."""
        sm = SecureMeasurement(encryption_key="test-key-compare")
        sig = _make_sig("test")
        envelope = sm.encrypt_signature(sig)

        current = _make_sig("test", {"response_key": "code", "latency_ms": 500})
        detector = DriftDetector(n_permutations=10)
        result = sm.secure_compare(envelope, current, detector)

        assert isinstance(result, ObfuscatedDriftResult)
        assert result.raw_distance_used is False
        # Ensure no raw vector data leaks
        result_dict = result.model_dump()
        for key, value in result_dict.items():
            if isinstance(value, list):
                assert key != "embedding_vector"

    def test_encryption_with_wrong_key_fails(self):
        """Decrypting with a different key must raise an integrity error."""
        sm1 = SecureMeasurement(encryption_key="key-alpha")
        sm2 = SecureMeasurement(encryption_key="key-beta")
        sig = _make_sig("test")
        envelope = sm1.encrypt_signature(sig)
        with pytest.raises(ValueError, match="Integrity check failed"):
            sm2.decrypt_signature(envelope)

    def test_noise_scale_controls_magnitude(self):
        """Larger noise_scale must produce larger average deviation."""
        sm = SecureMeasurement(encryption_key="test-key-scale")
        drift = _make_drift()

        deviations_small = []
        deviations_large = []

        # Run multiple times to average out randomness
        for _ in range(100):
            result_small = sm.obfuscate_drift(drift, noise_scale=0.01)
            result_large = sm.obfuscate_drift(drift, noise_scale=1.0)
            for k in drift.per_dimension_drift:
                deviations_small.append(
                    abs(result_small.obfuscated_dimensions[k] - drift.per_dimension_drift[k])
                )
                deviations_large.append(
                    abs(result_large.obfuscated_dimensions[k] - drift.per_dimension_drift[k])
                )

        avg_small = np.mean(deviations_small)
        avg_large = np.mean(deviations_large)
        assert avg_large > avg_small
