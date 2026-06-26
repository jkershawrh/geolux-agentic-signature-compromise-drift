from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_pipeline
from domain.identity import EnrollmentRequest

router = APIRouter()


class EnrollResponse(BaseModel):
    agent_id: str
    status: str
    display_name: str


class CertifyResponse(BaseModel):
    agent_id: str
    status: str
    all_checks_passed: bool
    self_consistency_passed: bool
    discriminability_passed: bool
    canary_pass_rate: float
    attack_detection_rate: float
    optimal_metric_count: int
    failure_reasons: list[str]


@router.post("/enroll", response_model=EnrollResponse)
def enroll_agent(request: EnrollmentRequest):
    pipeline = get_pipeline()
    agent = pipeline.enroll(request)
    return EnrollResponse(
        agent_id=agent.agent_id,
        status=agent.status.value,
        display_name=agent.display_name,
    )


@router.post("/{agent_id}/certify", response_model=CertifyResponse)
def certify_agent(agent_id: str):
    pipeline = get_pipeline()
    # Get the enrolled agent from repo
    from api.dependencies import get_repo

    repo = get_repo()
    agent = repo.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, f"Agent {agent_id} not found")
    report = pipeline.certify(agent)
    return CertifyResponse(
        agent_id=report.agent_id,
        status=report.status.value,
        all_checks_passed=report.all_checks_passed,
        self_consistency_passed=report.self_consistency_passed,
        discriminability_passed=report.discriminability_passed,
        canary_pass_rate=report.canary_pass_rate,
        attack_detection_rate=report.attack_detection_rate,
        optimal_metric_count=report.optimal_metric_count,
        failure_reasons=report.failure_reasons,
    )
