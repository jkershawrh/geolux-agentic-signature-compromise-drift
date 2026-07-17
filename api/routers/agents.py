from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_pipeline, get_repo
from db.repository import Repository
from domain.identity import EnrollmentRequest
from engine.behavior_profile import BehaviorProfiler
from engine.identity_pipeline import IdentityPipeline

router = APIRouter()


class EnrollResponse(BaseModel):
    agent_id: str
    status: str
    display_name: str


class MetricStatOut(BaseModel):
    metric_name: str
    dimension: str
    mean: float
    std: float


class ProfileResponse(BaseModel):
    agent_id: str
    n_runs: int
    n_metrics: int
    consistency_score: float
    per_dimension: dict[str, dict[str, float]]
    most_stable: list[MetricStatOut]
    most_variable: list[MetricStatOut]


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
def enroll_agent(
    request: EnrollmentRequest,
    pipeline: IdentityPipeline = Depends(get_pipeline),
):
    agent = pipeline.enroll(request)
    return EnrollResponse(
        agent_id=agent.agent_id,
        status=agent.status.value,
        display_name=agent.display_name,
    )


@router.post("/{agent_id}/certify", response_model=CertifyResponse)
def certify_agent(
    agent_id: str,
    repo: Repository = Depends(get_repo),
    pipeline: IdentityPipeline = Depends(get_pipeline),
):
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


@router.get("/{agent_id}/profile", response_model=ProfileResponse)
def get_behavior_profile(agent_id: str, repo: Repository = Depends(get_repo)):
    """Interpretable behavioral profile from the agent's stored metrics."""
    agent = repo.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, f"Agent {agent_id} not found")
    metrics = repo.get_metrics_for_agent(agent_id)
    profile = BehaviorProfiler().profile(agent_id, metrics)
    return ProfileResponse(
        agent_id=profile.agent_id,
        n_runs=profile.n_runs,
        n_metrics=profile.n_metrics,
        consistency_score=profile.consistency_score,
        per_dimension=profile.per_dimension,
        most_stable=[
            MetricStatOut(metric_name=s.metric_name, dimension=s.dimension,
                          mean=round(s.mean, 4), std=round(s.std, 4))
            for s in profile.most_stable
        ],
        most_variable=[
            MetricStatOut(metric_name=s.metric_name, dimension=s.dimension,
                          mean=round(s.mean, 4), std=round(s.std, 4))
            for s in profile.most_variable
        ],
    )
