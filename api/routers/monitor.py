from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_pipeline, get_repo
from db.repository import Repository
from domain.enums import AgentStatus, RunStatus
from domain.identity import EnforcementAction, MonitoringPolicy
from domain.models import ControlledRun
from engine.identity_pipeline import IdentityPipeline

router = APIRouter()


class CheckRequest(BaseModel):
    prompt: str
    response_text: str
    model_id: str = "default"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    policy: MonitoringPolicy = MonitoringPolicy.GRADUATED


class CheckResponse(BaseModel):
    agent_id: str
    drift_score: float
    action: str
    event_type: str
    strike_count: int
    baseline_integrity: str
    per_dimension_drift: dict[str, float]
    top_shifted_dimensions: list[str]


class StatusResponse(BaseModel):
    agent_id: str
    status: str
    strike_count: int


@router.post("/{agent_id}/check", response_model=CheckResponse)
def check_drift(
    agent_id: str,
    request: CheckRequest,
    repo: Repository = Depends(get_repo),
    pipeline: IdentityPipeline = Depends(get_pipeline),
):
    agent = repo.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, f"Agent {agent_id} not found")

    if agent.status == AgentStatus.COMPROMISED:
        raise HTTPException(403, f"Agent {agent_id} is suspended (COMPROMISED)")

    baseline = repo.get_baseline_signature(agent_id)
    if not baseline:
        raise HTTPException(400, f"Agent {agent_id} has no baseline signature")

    # Refuse to monitor against a baseline that fails its sealed-envelope
    # check — a tampered baseline would let drift go undetected.
    integrity_ok, integrity_reason = pipeline.verify_baseline_integrity(
        baseline, agent_status=agent.status
    )
    if not integrity_ok:
        raise HTTPException(409, f"baseline integrity check failed: {integrity_reason}")

    # Create a ControlledRun from the request
    run = ControlledRun(
        agent_id=agent_id,
        scenario_id="api_check",
        prompt_text=request.prompt,
        response_text=request.response_text,
        model_id=request.model_id,
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
        latency_ms=request.latency_ms,
        status=RunStatus.COMPLETED,
    )

    # Extract metrics once, reuse for both monitoring and breakdown
    metrics = pipeline.extract_metrics(run)
    event = pipeline.monitor(agent, run, baseline)

    breakdown = pipeline.drift_breakdown(run, baseline, pre_extracted_metrics=metrics)
    top_shifted = sorted(breakdown, key=breakdown.get, reverse=True)[:3]

    # Run enforcement against the persisted strike count so the graduated
    # (3-strike) policy actually escalates across requests.
    strike_count = repo.get_strike_count(agent_id)
    alert = pipeline.respond(agent, event, request.policy, strike_count)
    action = event.action_taken.value
    if alert is not None:
        action = alert.action_taken.value
        new_strikes = alert.strike_count
        if new_strikes != strike_count:
            repo.increment_strike_count(agent_id, new_strikes - strike_count)
        strike_count = new_strikes
        if alert.action_taken == EnforcementAction.SUSPEND:
            repo.update_agent_status(agent_id, AgentStatus.COMPROMISED)

    return CheckResponse(
        agent_id=agent_id,
        drift_score=event.drift_score,
        action=action,
        event_type=event.event_type,
        strike_count=strike_count,
        baseline_integrity=integrity_reason,
        per_dimension_drift={k: round(v, 6) for k, v in breakdown.items()},
        top_shifted_dimensions=top_shifted,
    )


@router.get("/{agent_id}/status", response_model=StatusResponse)
def get_status(agent_id: str, repo: Repository = Depends(get_repo)):
    agent = repo.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, f"Agent {agent_id} not found")
    return StatusResponse(
        agent_id=agent_id,
        status=agent.status.value,
        strike_count=repo.get_strike_count(agent_id),
    )
