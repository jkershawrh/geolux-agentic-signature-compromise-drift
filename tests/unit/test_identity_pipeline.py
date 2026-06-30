"""Smoke tests for the identity pipeline orchestrator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import AgentStatus
from domain.identity import EnrollmentRequest
from engine.identity_pipeline import IdentityPipeline


def test_enroll_sets_enrolled_status():
    pipeline = IdentityPipeline(
        MockInferenceAdapter(), DefaultMetricExtractor(),
    )
    agent = pipeline.enroll(EnrollmentRequest(
        agent_id="pipe-001",
        display_name="Pipeline Agent",
        model_id="mock-model",
        system_prompt="You are helpful.",
    ))
    assert agent.status == AgentStatus.ENROLLED
    assert agent.agent_id == "pipe-001"


def test_enroll_persists_when_repository_provided(repository):
    pipeline = IdentityPipeline(
        MockInferenceAdapter(), DefaultMetricExtractor(), repository=repository,
    )
    agent = pipeline.enroll(EnrollmentRequest(
        agent_id="pipe-002",
        display_name="Persisted Agent",
        model_id="mock-model",
        system_prompt="You are helpful.",
    ))
    loaded = repository.get_agent("pipe-002")
    assert loaded is not None
    assert loaded.agent_id == agent.agent_id
