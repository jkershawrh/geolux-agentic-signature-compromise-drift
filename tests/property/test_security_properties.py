from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from domain.enums import DriftCategory
from domain.geometry import DriftMeasurement
from engine.secure_measurement import SecureMeasurement


def _vector_strategy(min_size=3, max_size=20):
    """Strategy that generates non-empty lists of finite floats."""
    return st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=min_size,
        max_size=max_size,
    )


class TestSecurityProperties:
    @given(vector=_vector_strategy())
    @settings(max_examples=50)
    def test_encrypt_decrypt_roundtrip_always_valid(self, vector):
        """Encrypt/decrypt must always recover the original vector."""
        from domain.enums import SignatureType
        from domain.geometry import GeometricSignature

        sm = SecureMeasurement(encryption_key="prop-test-key")
        sig = GeometricSignature(
            agent_id="prop-agent",
            signature_type=SignatureType.BASELINE,
            embedding_vector=vector,
            embedding_dimension=len(vector),
            manifold_coordinates=[0.5, 0.3],
            metric_snapshot={"test": 0.5},
            run_ids=["run-001"],
            num_runs=1,
            computation_method="test",
        )
        envelope = sm.encrypt_signature(sig)
        recovered = sm.decrypt_signature(envelope)
        assert recovered == vector

    @given(vector=_vector_strategy())
    @settings(max_examples=50)
    def test_commitment_hash_deterministic(self, vector):
        """Same vector must always produce the same commitment hash."""
        h1 = SecureMeasurement.compute_commitment_hash(vector)
        h2 = SecureMeasurement.compute_commitment_hash(vector)
        assert h1 == h2

    @given(
        noise_scale=st.floats(min_value=0.001, max_value=0.1, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_obfuscated_noise_bounded_by_scale(self, noise_scale):
        """Obfuscated values must remain non-negative and bounded."""
        sm = SecureMeasurement(encryption_key="prop-test-noise")
        drift = DriftMeasurement(
            agent_id="prop-agent",
            baseline_signature_id="s1",
            current_signature_id="s2",
            geodesic_distance=0.5,
            euclidean_distance=0.4,
            cosine_similarity=0.85,
            drift_category=DriftCategory.SEMANTIC,
            drift_magnitude=0.3,
            per_dimension_drift={
                "response_structure": 0.12,
                "reasoning_pattern": 0.15,
            },
            is_significant=True,
            p_value=0.03,
            compromise_probability=0.4,
        )
        result = sm.obfuscate_drift(drift, noise_scale=noise_scale)
        for key, value in result.obfuscated_dimensions.items():
            assert value >= 0.0, f"Obfuscated value for {key} is negative: {value}"
