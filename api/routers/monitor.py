from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.dependencies import get_pipeline, get_repo
from domain.enums import RunStatus
from domain.identity import MonitoringPolicy
from domain.models import ControlledRun

router = APIRouter()


class CheckRequest(BaseModel):
    prompt: str
    response_text: str
    model_id: str = "default"
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


class CheckResponse(BaseModel):
    agent_id: str
    drift_score: float
    action: str
    event_type: str


class StatusResponse(BaseModel):
    agent_id: str
    status: str
    strike_count: int


@router.post("/{agent_id}/check", response_model=CheckResponse)
def check_drift(agent_id: str, request: CheckRequest):
    pipeline = get_pipeline()
    repo = get_repo()
    agent = repo.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, f"Agent {agent_id} not found")

    baseline = repo.get_baseline_signature(agent_id)
    if not baseline:
        raise HTTPException(400, f"Agent {agent_id} has no baseline signature")

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

    event = pipeline.monitor(agent, run, baseline)
    return CheckResponse(
        agent_id=agent_id,
        drift_score=event.drift_score,
        action=event.action_taken.value,
        event_type=event.event_type,
    )


@router.get("/{agent_id}/status", response_model=StatusResponse)
def get_status(agent_id: str):
    repo = get_repo()
    agent = repo.get_agent(agent_id)
    if not agent:
        raise HTTPException(404, f"Agent {agent_id} not found")
    return StatusResponse(
        agent_id=agent_id,
        status=agent.status.value,
        strike_count=0,
    )
