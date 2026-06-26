from typing import Protocol

from domain.metrics import MetricMeasurement
from domain.models import AgentProfile, ControlledRun


class InferenceAdapter(Protocol):
    def execute(self, agent: AgentProfile, prompt: str) -> ControlledRun: ...


class MetricExtractor(Protocol):
    def extract(self, run: ControlledRun) -> list[MetricMeasurement]: ...


class ConversationalAdapter(Protocol):
    def execute_turn(self, agent: AgentProfile, messages: list[dict[str, str]]) -> ControlledRun: ...
