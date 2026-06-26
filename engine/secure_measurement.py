from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np

from domain.geometry import DriftMeasurement, GeometricSignature
from domain.security import (
    MeasurementAuditRecord,
    ObfuscatedDriftResult,
    SecureSignatureEnvelope,
)
from engine.drift_detector import DriftDetector


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SimpleEncryptor:
    """XOR cipher with HMAC integrity — stdlib-only, no Rust compiler needed.

    Suitable for demonstrating the security architecture pattern in a
    research prototype.  NOT intended for production use.
    """

    def __init__(self, key: bytes):
        self._key = key

    def encrypt(self, data: bytes) -> str:
        """XOR with repeating key, base64 encode, append HMAC."""
        key_stream = (self._key * (len(data) // len(self._key) + 1))[: len(data)]
        encrypted = bytes(a ^ b for a, b in zip(data, key_stream))
        mac = hmac.new(self._key, encrypted, hashlib.sha256).hexdigest()
        return base64.b64encode(encrypted).decode() + ":" + mac

    def decrypt(self, token: str) -> bytes:
        """Verify HMAC, then XOR decrypt."""
        encoded, mac = token.rsplit(":", 1)
        encrypted = base64.b64decode(encoded)
        expected_mac = hmac.new(self._key, encrypted, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac, expected_mac):
            raise ValueError("Integrity check failed — signature may be tampered")
        key_stream = (self._key * (len(encrypted) // len(self._key) + 1))[
            : len(encrypted)
        ]
        return bytes(a ^ b for a, b in zip(encrypted, key_stream))


class SecureMeasurement:
    """Security layer wrapping drift detection with encryption, obfuscation,
    and access-controlled comparisons.

    Encrypts geometric signature vectors at rest, computes SHA-256 commitment
    hashes for tamper detection, and adds calibrated noise to drift results
    before exposing them externally.
    """

    def __init__(self, encryption_key: str | None = None):
        raw_key = encryption_key or os.environ.get("ASC_ENCRYPTION_KEY")
        if raw_key is None:
            raw_key = base64.b64encode(os.urandom(32)).decode()
            warnings.warn(
                "No encryption key provided — generated ephemeral key. "
                "Set ASC_ENCRYPTION_KEY for production use.",
                stacklevel=2,
            )
        key_bytes = hashlib.pbkdf2_hmac(
            "sha256", raw_key.encode(), b"asc-salt", iterations=100_000
        )
        self._encryptor = SimpleEncryptor(key_bytes)
        self._audit_log: list[MeasurementAuditRecord] = []

    # ------------------------------------------------------------------
    # Encryption
    # ------------------------------------------------------------------

    def encrypt_signature(self, signature: GeometricSignature) -> SecureSignatureEnvelope:
        """Encrypt the embedding vector and produce a tamper-evident envelope."""
        raw_bytes = json.dumps(signature.embedding_vector).encode()
        encrypted = self._encryptor.encrypt(raw_bytes)
        commitment = self.compute_commitment_hash(signature.embedding_vector)

        envelope = SecureSignatureEnvelope(
            agent_id=signature.agent_id,
            signature_id=signature.signature_id,
            encrypted_vector=encrypted,
            commitment_hash=commitment,
        )

        self._record_audit("encrypt", signature.agent_id, True)
        return envelope

    def decrypt_signature(self, envelope: SecureSignatureEnvelope) -> list[float]:
        """Decrypt the envelope and verify the commitment hash."""
        try:
            raw_bytes = self._encryptor.decrypt(envelope.encrypted_vector)
        except ValueError:
            self._record_audit("decrypt", envelope.agent_id, False,
                               {"reason": "integrity_check_failed"})
            raise

        vector: list[float] = json.loads(raw_bytes)

        if not self.verify_commitment(envelope, vector):
            self._record_audit("decrypt", envelope.agent_id, False,
                               {"reason": "commitment_mismatch"})
            raise ValueError("Commitment hash mismatch — vector may be tampered")

        self._record_audit("decrypt", envelope.agent_id, True)
        return vector

    # ------------------------------------------------------------------
    # Commitment hashes
    # ------------------------------------------------------------------

    def verify_commitment(self, envelope: SecureSignatureEnvelope, raw_vector: list[float]) -> bool:
        """Check that the SHA-256 of *raw_vector* matches the envelope hash."""
        computed = self.compute_commitment_hash(raw_vector)
        result = hmac.compare_digest(computed, envelope.commitment_hash)
        self._record_audit("verify_commitment", envelope.agent_id, result)
        return result

    @staticmethod
    def compute_commitment_hash(vector: list[float]) -> str:
        """SHA-256 of the JSON-serialized vector."""
        serialized = json.dumps(vector).encode()
        return hashlib.sha256(serialized).hexdigest()

    # ------------------------------------------------------------------
    # Drift obfuscation
    # ------------------------------------------------------------------

    def obfuscate_drift(
        self,
        drift_measurement: DriftMeasurement,
        noise_scale: float = 0.05,
    ) -> ObfuscatedDriftResult:
        """Add calibrated noise to per-dimension drift and classify severity.

        The raw geodesic distance is used *internally* for severity
        classification but is **not** exposed in the result.
        """
        noisy_dims = self._add_calibrated_noise(
            drift_measurement.per_dimension_drift, noise_scale
        )
        severity = self._classify_severity(drift_measurement.geodesic_distance)
        is_drifted = severity != "none"

        return ObfuscatedDriftResult(
            agent_id=drift_measurement.agent_id,
            is_drifted=is_drifted,
            severity=severity,
            obfuscated_dimensions=noisy_dims,
            raw_distance_used=False,
        )

    # ------------------------------------------------------------------
    # Secure comparison (end-to-end)
    # ------------------------------------------------------------------

    def secure_compare(
        self,
        baseline_envelope: SecureSignatureEnvelope,
        current_signature: GeometricSignature,
        drift_detector: DriftDetector,
    ) -> ObfuscatedDriftResult:
        """Decrypt baseline, run drift detection, and return obfuscated result."""
        baseline_vector = self.decrypt_signature(baseline_envelope)

        # Reconstruct a minimal baseline GeometricSignature for the detector
        baseline_sig = GeometricSignature(
            signature_id=baseline_envelope.signature_id,
            agent_id=baseline_envelope.agent_id,
            signature_type=current_signature.signature_type,
            embedding_vector=baseline_vector,
            embedding_dimension=len(baseline_vector),
            manifold_coordinates=current_signature.manifold_coordinates,
            metric_snapshot=current_signature.metric_snapshot,
            run_ids=current_signature.run_ids,
            num_runs=current_signature.num_runs,
            computation_method=current_signature.computation_method,
        )

        drift = drift_detector.detect(baseline_sig, current_signature)
        result = self.obfuscate_drift(drift)

        self._record_audit("compare", current_signature.agent_id, True, {
            "severity": result.severity,
            "is_drifted": result.is_drifted,
        })

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_calibrated_noise(
        self, values_dict: dict[str, float], noise_scale: float
    ) -> dict[str, float]:
        """Add Gaussian noise proportional to each value."""
        rng = np.random.default_rng()
        noisy: dict[str, float] = {}
        for key, value in values_dict.items():
            noise = rng.normal(0, noise_scale * max(abs(value), 0.001))
            clamped = max(0.0, value + noise)
            upper = max(value * 2, 0.01)
            noisy[key] = float(min(clamped, upper))
        return noisy

    @staticmethod
    def _classify_severity(geodesic_distance: float) -> str:
        if geodesic_distance < 0.3:
            return "none"
        if geodesic_distance < 1.0:
            return "warning"
        return "critical"

    def _record_audit(
        self, operation: str, agent_id: str, success: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._audit_log.append(
            MeasurementAuditRecord(
                operation=operation,
                agent_id=agent_id,
                success=success,
                details=details or {},
            )
        )

    @property
    def audit_log(self) -> list[MeasurementAuditRecord]:
        return list(self._audit_log)
