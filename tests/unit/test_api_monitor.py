"""API tests for the monitor router: persisted strikes and graduated enforcement."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from fastapi.testclient import TestClient

import api.dependencies as deps
from api.app import create_app
from domain.enums import SignatureType
from domain.geometry import GeometricSignature


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient backed by a fresh temp database."""
    monkeypatch.setenv("ASC_DATABASE_PATH", str(tmp_path / "test_asc.db"))
    monkeypatch.setattr(deps, "_session_factory", None)
    with TestClient(create_app()) as c:
        yield c
    deps._session_factory = None


def _enroll(client: TestClient, agent_id: str = "api-test-agent") -> str:
    resp = client.post(
        "/agents/enroll",
        json={
            "agent_id": agent_id,
            "display_name": "API Test Agent",
            "model_id": "mock-model",
            "system_prompt": "You are a test agent.",
        },
    )
    assert resp.status_code == 200, resp.text
    return agent_id


def _save_baseline(agent_id: str, n_metrics: int = 36) -> None:
    """Persist a far-away baseline so every check registers drift."""
    session = deps._get_session_factory()()
    try:
        from db.repository import Repository

        repo = Repository(session)
        repo.save_signature(
            GeometricSignature(
                agent_id=agent_id,
                signature_type=SignatureType.BASELINE,
                embedding_vector=[10.0] * n_metrics,
                embedding_dimension=n_metrics,
                manifold_coordinates=[0.0, 0.0],
                metric_snapshot={},
                run_ids=["baseline-run"],
                num_runs=1,
                computation_method="pca",
            )
        )
    finally:
        session.close()


CHECK_PAYLOAD = {
    "prompt": "What is drift?",
    "response_text": "A short response.",
    "model_id": "mock-model",
    "input_tokens": 10,
    "output_tokens": 5,
    "latency_ms": 100,
}


def test_status_unknown_agent_404(client):
    assert client.get("/monitor/nope/status").status_code == 404


def test_check_without_baseline_400(client):
    agent_id = _enroll(client)
    resp = client.post(f"/monitor/{agent_id}/check", json=CHECK_PAYLOAD)
    assert resp.status_code == 400


def test_strikes_persist_and_escalate(client):
    agent_id = _enroll(client)
    _save_baseline(agent_id)

    actions = []
    for _ in range(3):
        resp = client.post(f"/monitor/{agent_id}/check", json=CHECK_PAYLOAD)
        assert resp.status_code == 200, resp.text
        actions.append(resp.json())

    # Graduated policy: warning -> throttle -> suspend across requests
    assert [a["strike_count"] for a in actions] == [1, 2, 3]
    assert [a["action"] for a in actions] == ["warning", "throttle", "suspend"]

    # Strike count survives into the status endpoint (persisted, not in-memory)
    status = client.get(f"/monitor/{agent_id}/status").json()
    assert status["strike_count"] == 3
    assert status["status"] == "compromised"


def test_check_reports_per_dimension_drift(client):
    agent_id = _enroll(client)
    _save_baseline(agent_id)

    body = client.post(f"/monitor/{agent_id}/check", json=CHECK_PAYLOAD).json()
    breakdown = body["per_dimension_drift"]
    assert len(breakdown) == 9  # one entry per behavioral dimension
    assert all(v >= 0 for v in breakdown.values())
    # Top shifted dimensions are the argmax of the breakdown
    top = body["top_shifted_dimensions"]
    assert len(top) == 3
    assert breakdown[top[0]] == max(breakdown.values())
    # Baseline was saved without an envelope -> integrity note, not a failure
    assert "no envelope" in body["baseline_integrity"]
