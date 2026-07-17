"""Tests for behavioral profiling, provenance chains, baseline sealing, API auth."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import RealisticMockAdapter
from db.database import create_db_engine, get_session_factory, init_db
from db.repository import Repository
from domain.enums import MetricDimension, RunStatus, SignatureType
from domain.geometry import GeometricSignature
from domain.metrics import MetricMeasurement
from domain.models import ControlledRun
from engine.behavior_profile import BehaviorProfiler
from engine.identity_pipeline import IdentityPipeline
from engine.provenance import ProvenanceSigner
from engine.secure_measurement import SecureMeasurement


def _run(agent_id: str = "a1", response: str = "A helpful answer.") -> ControlledRun:
    return ControlledRun(
        agent_id=agent_id, scenario_id="test", prompt_text="Q?",
        response_text=response, model_id="mock", status=RunStatus.COMPLETED,
    )


def _metric(run_id: str, name: str, value: float,
            dim: MetricDimension = MetricDimension.RESPONSE_STRUCTURE) -> MetricMeasurement:
    return MetricMeasurement(
        metric_id=str(uuid.uuid4()), run_id=run_id, agent_id="a1",
        dimension=dim, metric_name=name, value=value, normalized_value=value,
    )


# ---------------------------------------------------------------------------
# Behavioral analysis: BehaviorProfiler
# ---------------------------------------------------------------------------


class TestBehaviorProfiler:
    def test_profile_aggregates_across_runs(self):
        metrics = [
            _metric("r1", "avg_response_length", 0.5),
            _metric("r2", "avg_response_length", 0.5),
            _metric("r1", "vocabulary_diversity", 0.2, MetricDimension.SEMANTIC_CONSISTENCY),
            _metric("r2", "vocabulary_diversity", 0.8, MetricDimension.SEMANTIC_CONSISTENCY),
        ]
        profile = BehaviorProfiler(top_k=2).profile("a1", metrics)
        assert profile.n_runs == 2
        assert profile.n_metrics == 2
        # Constant metric is the most stable; the volatile one the most variable
        assert profile.most_stable[0].metric_name == "avg_response_length"
        assert profile.most_variable[0].metric_name == "vocabulary_diversity"
        assert set(profile.per_dimension) == {"response_structure", "semantic_consistency"}
        assert 0.0 < profile.consistency_score <= 1.0

    def test_perfectly_consistent_agent_scores_one(self):
        metrics = [_metric(r, "m", 0.4) for r in ("r1", "r2", "r3")]
        profile = BehaviorProfiler().profile("a1", metrics)
        assert profile.consistency_score == 1.0

    def test_empty_metrics_yield_empty_profile(self):
        profile = BehaviorProfiler().profile("a1", [])
        assert profile.n_runs == 0
        assert profile.consistency_score == 0.0


# ---------------------------------------------------------------------------
# Security: provenance chains
# ---------------------------------------------------------------------------


class TestProvenanceChain:
    def test_chain_signs_and_verifies(self):
        signer = ProvenanceSigner(key="prov-key")
        for i in range(3):
            signer.sign_run(_run(response=f"answer {i}"))
        chain = signer.chain_for("a1")
        assert [r.sequence for r in chain] == [0, 1, 2]
        result = signer.verify_chain(chain)
        assert result.valid
        assert result.records_checked == 3

    def test_tampered_record_breaks_chain(self):
        signer = ProvenanceSigner(key="prov-key")
        for i in range(3):
            signer.sign_run(_run(response=f"answer {i}"))
        chain = signer.chain_for("a1")
        chain[1].response_hash = "0" * 64  # rewrite history
        result = signer.verify_chain(chain)
        assert not result.valid
        assert result.first_invalid_sequence == 1
        assert "signature mismatch" in result.reason

    def test_reordering_detected(self):
        signer = ProvenanceSigner(key="prov-key")
        for i in range(3):
            signer.sign_run(_run(response=f"answer {i}"))
        chain = signer.chain_for("a1")
        chain[0], chain[1] = chain[1], chain[0]
        assert not signer.verify_chain(chain).valid

    def test_response_text_mismatch_detected(self):
        signer = ProvenanceSigner(key="prov-key")
        record = signer.sign_run(_run(response="the original response"))
        result = signer.verify_chain(
            signer.chain_for("a1"), responses={record.run_id: "a swapped response"}
        )
        assert not result.valid
        assert "does not match signed hash" in result.reason

    def test_wrong_key_fails_verification(self):
        signer = ProvenanceSigner(key="prov-key")
        signer.sign_run(_run())
        other = ProvenanceSigner(key="different-key")
        assert not other.verify_chain(signer.chain_for("a1")).valid


# ---------------------------------------------------------------------------
# Security: sealed baselines
# ---------------------------------------------------------------------------


class TestBaselineSealing:
    @pytest.fixture()
    def repo(self):
        engine = create_db_engine(":memory:")
        init_db(engine)
        session = get_session_factory(engine)()
        yield Repository(session)
        session.close()

    def _pipeline(self, repo):
        return IdentityPipeline(
            adapter=RealisticMockAdapter(profile="balanced"),
            extractor=DefaultMetricExtractor(),
            repository=repo,
            secure=SecureMeasurement(encryption_key="seal-key"),
        )

    def _baseline(self, n: int = 36) -> GeometricSignature:
        return GeometricSignature(
            agent_id="a1", signature_type=SignatureType.BASELINE,
            embedding_vector=[0.5] * n, embedding_dimension=n,
            manifold_coordinates=[0.0, 0.0], metric_snapshot={},
            run_ids=["r1"], num_runs=1, computation_method="pca",
        )

    def test_sealed_baseline_verifies(self, repo):
        pipeline = self._pipeline(repo)
        baseline = self._baseline()
        repo.save_signature(baseline)
        repo.save_envelope(pipeline._secure.encrypt_signature(baseline))
        ok, reason = pipeline.verify_baseline_integrity(baseline)
        assert ok and "verified" in reason

    def test_tampered_baseline_detected(self, repo):
        pipeline = self._pipeline(repo)
        baseline = self._baseline()
        repo.save_signature(baseline)
        repo.save_envelope(pipeline._secure.encrypt_signature(baseline))
        tampered = baseline.model_copy(
            update={"embedding_vector": [0.9] * baseline.embedding_dimension}
        )
        ok, reason = pipeline.verify_baseline_integrity(tampered)
        assert not ok and "tampered" in reason

    def test_unsealed_baseline_passes_with_note(self, repo):
        pipeline = self._pipeline(repo)
        baseline = self._baseline()
        repo.save_signature(baseline)
        ok, reason = pipeline.verify_baseline_integrity(baseline)
        assert ok and "no envelope" in reason


# ---------------------------------------------------------------------------
# Security: API-key authentication
# ---------------------------------------------------------------------------


class TestApiAuth:
    @pytest.fixture()
    def client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        import api.dependencies as deps
        from api.app import create_app

        monkeypatch.setenv("ASC_DATABASE_PATH", str(tmp_path / "auth_test.db"))
        monkeypatch.setenv("ASC_API_KEY", "sekrit")
        monkeypatch.setattr(deps, "_session_factory", None)
        with TestClient(create_app()) as c:
            yield c
        deps._session_factory = None

    def test_requests_without_key_rejected(self, client):
        assert client.get("/agents/x/profile").status_code == 401
        assert client.get("/monitor/x/status").status_code == 401

    def test_requests_with_key_pass_auth(self, client):
        resp = client.get("/agents/x/profile", headers={"X-API-Key": "sekrit"})
        assert resp.status_code == 404  # authenticated; agent simply doesn't exist

    def test_wrong_key_rejected(self, client):
        resp = client.get("/agents/x/profile", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_health_stays_open(self, client):
        assert client.get("/health").status_code == 200
