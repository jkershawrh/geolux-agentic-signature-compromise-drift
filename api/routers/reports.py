from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any

from api.dependencies import get_repo

router = APIRouter()


class CertificationSummary(BaseModel):
    certification_id: str
    agent_id: str
    status: str
    canary_pass_rate: float
    attack_detection_rate: float
    all_passed: bool


class DriftSummary(BaseModel):
    event_id: str
    agent_id: str
    drift_score: float
    action_taken: str
    event_type: str


@router.get("/{agent_id}/certifications", response_model=list[CertificationSummary])
def get_certifications(agent_id: str):
    repo = get_repo()
    certs = repo.get_certifications_for_agent(agent_id)
    return [
        CertificationSummary(
            certification_id=c.certification_id,
            agent_id=c.agent_id,
            status=c.status.value,
            canary_pass_rate=c.canary_pass_rate,
            attack_detection_rate=c.attack_detection_rate,
            all_passed=c.all_checks_passed,
        )
        for c in certs
    ]


@router.get("/{agent_id}/drift", response_model=list[DriftSummary])
def get_drift_events(agent_id: str, limit: int = 50):
    repo = get_repo()
    events = repo.get_monitoring_events(agent_id, limit=limit)
    return [
        DriftSummary(
            event_id=e.event_id,
            agent_id=e.agent_id,
            drift_score=e.drift_score,
            action_taken=e.action_taken.value,
            event_type=e.event_type,
        )
        for e in events
    ]
