from domain.enums import AgentStatus, RunStatus

VALID_AGENT_TRANSITIONS: set[tuple[AgentStatus, AgentStatus]] = {
    (AgentStatus.BASELINE_PENDING, AgentStatus.ACTIVE),
    (AgentStatus.ENROLLED, AgentStatus.CERTIFIED),
    (AgentStatus.ENROLLED, AgentStatus.ARCHIVED),
    (AgentStatus.CERTIFIED, AgentStatus.ACTIVE),
    (AgentStatus.ACTIVE, AgentStatus.COMPROMISED),
    (AgentStatus.COMPROMISED, AgentStatus.RECOVERED),
    (AgentStatus.RECOVERED, AgentStatus.ACTIVE),
    (AgentStatus.ACTIVE, AgentStatus.ARCHIVED),
    (AgentStatus.RECOVERED, AgentStatus.ARCHIVED),
}

VALID_RUN_TRANSITIONS: set[tuple[RunStatus, RunStatus]] = {
    (RunStatus.PENDING, RunStatus.RUNNING),
    (RunStatus.RUNNING, RunStatus.COMPLETED),
    (RunStatus.RUNNING, RunStatus.FAILED),
}


def is_valid_agent_transition(from_status: AgentStatus, to_status: AgentStatus) -> bool:
    return (from_status, to_status) in VALID_AGENT_TRANSITIONS


def is_valid_run_transition(from_status: RunStatus, to_status: RunStatus) -> bool:
    return (from_status, to_status) in VALID_RUN_TRANSITIONS
