"""Response provenance: HMAC-signed, hash-chained records of agent output.

Turns "which agent produced this text?" from a statistical inference into a
verification. Each response is recorded as an HMAC-SHA256 signature over
``(agent_id, sequence, prompt_hash, response_hash, previous_signature)``,
chained per agent. Anyone holding the key can later verify that a transcript
is complete, ordered, and unmodified; tampering with any response (or
deleting/reordering records) breaks the chain from that point forward.

Scope: this proves integrity and attribution of the *recorded* traffic from
the point of observation onward. It does not prove which model produced the
text — pair it with workload attestation for that. Sign at the inference
gateway (where responses are produced), not after the fact.

Key handling mirrors SecureMeasurement: pass a key or set
``ASC_PROVENANCE_KEY``; an ephemeral key is generated (with a warning)
otherwise, which is fine for tests and useless for real provenance.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from domain.models import ControlledRun

_GENESIS = "genesis"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ProvenanceRecord:
    """One link in an agent's response provenance chain."""

    agent_id: str
    sequence: int
    run_id: str
    prompt_hash: str
    response_hash: str
    prev_signature: str
    signature: str
    created_at: datetime = field(default_factory=_utcnow)


@dataclass
class ChainVerification:
    """Result of verifying a provenance chain."""

    valid: bool
    records_checked: int
    first_invalid_sequence: Optional[int] = None
    reason: str = ""


class ProvenanceSigner:
    """Sign and verify hash-chained response provenance records."""

    def __init__(self, key: Optional[str] = None):
        raw_key = key or os.environ.get("ASC_PROVENANCE_KEY")
        if raw_key is None:
            raw_key = base64.b64encode(os.urandom(32)).decode()
            warnings.warn(
                "No provenance key provided — generated ephemeral key. "
                "Set ASC_PROVENANCE_KEY so chains verify across processes.",
                stacklevel=2,
            )
        self._key = hashlib.pbkdf2_hmac(
            "sha256", raw_key.encode(), b"asc-provenance", iterations=100_000
        )
        self._chains: dict[str, list[ProvenanceRecord]] = {}

    # ------------------------------------------------------------------
    # Signing
    # ------------------------------------------------------------------

    def sign_run(self, run: ControlledRun) -> ProvenanceRecord:
        """Append a signed record for this run to the agent's chain."""
        chain = self._chains.setdefault(run.agent_id, [])
        prev_signature = chain[-1].signature if chain else _GENESIS
        sequence = len(chain)
        prompt_hash = _sha256(run.prompt_text)
        response_hash = _sha256(run.response_text)

        record = ProvenanceRecord(
            agent_id=run.agent_id,
            sequence=sequence,
            run_id=run.run_id,
            prompt_hash=prompt_hash,
            response_hash=response_hash,
            prev_signature=prev_signature,
            signature=self._sign(
                run.agent_id, sequence, prompt_hash, response_hash, prev_signature
            ),
        )
        chain.append(record)
        return record

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_chain(
        self,
        records: list[ProvenanceRecord],
        responses: Optional[dict[str, str]] = None,
    ) -> ChainVerification:
        """Verify an agent's provenance chain.

        Checks per record: sequence continuity, chain linkage to the
        previous signature, and the HMAC itself. If *responses* is given
        (run_id -> response_text), also checks that each stored response
        still matches its signed hash.
        """
        prev_signature = _GENESIS
        for i, record in enumerate(records):
            if record.sequence != i:
                return ChainVerification(
                    valid=False, records_checked=i,
                    first_invalid_sequence=record.sequence,
                    reason="sequence gap or reordering",
                )
            if record.prev_signature != prev_signature:
                return ChainVerification(
                    valid=False, records_checked=i,
                    first_invalid_sequence=record.sequence,
                    reason="chain linkage broken",
                )
            expected = self._sign(
                record.agent_id, record.sequence,
                record.prompt_hash, record.response_hash,
                record.prev_signature,
            )
            if not hmac.compare_digest(expected, record.signature):
                return ChainVerification(
                    valid=False, records_checked=i,
                    first_invalid_sequence=record.sequence,
                    reason="signature mismatch (record tampered)",
                )
            if responses is not None and record.run_id in responses:
                if _sha256(responses[record.run_id]) != record.response_hash:
                    return ChainVerification(
                        valid=False, records_checked=i,
                        first_invalid_sequence=record.sequence,
                        reason="response text does not match signed hash",
                    )
            prev_signature = record.signature

        return ChainVerification(valid=True, records_checked=len(records))

    def chain_for(self, agent_id: str) -> list[ProvenanceRecord]:
        return list(self._chains.get(agent_id, []))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sign(
        self,
        agent_id: str,
        sequence: int,
        prompt_hash: str,
        response_hash: str,
        prev_signature: str,
    ) -> str:
        payload = "\x1f".join(
            [agent_id, str(sequence), prompt_hash, response_hash, prev_signature]
        ).encode("utf-8")
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()
