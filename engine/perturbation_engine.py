from __future__ import annotations

import copy
import importlib
from typing import Any, Optional

from domain.models import AgentProfile


class PerturbationEngine:
    """Introduces controlled changes to agents for measuring signature drift.

    Each perturbation type modifies the agent or its inputs in a specific way,
    enabling measurement of exactly one variable at a time.
    """

    def apply_scenario(
        self, agent: AgentProfile, scenario_id: str
    ) -> tuple[AgentProfile, list[str], dict[str, Any]]:
        """Apply a scenario's perturbation to an agent.

        Returns (modified_agent, modified_prompts, perturbation_record).
        """
        scenario = self._load_scenario(scenario_id)
        prompts = list(scenario.PROMPTS)
        perturbation = scenario.PERTURBATION

        if perturbation is None:
            return agent, prompts, {}

        modified_agent = copy.deepcopy(agent)
        perturbation_record = dict(perturbation)

        ptype = perturbation.get("type", "")

        if ptype == "prompt_injection":
            override = perturbation.get("system_prompt_override", "")
            if override:
                modified_agent.system_prompt = override
                modified_agent.system_prompt_hash = ""
                modified_agent.model_post_init(None)

        elif ptype == "model_swap":
            swapped = perturbation.get("swapped_model", "")
            if swapped:
                modified_agent.model_id = swapped

        elif ptype == "parameter_drift":
            style = perturbation.get("style_override", "")
            if style:
                original_prompt = modified_agent.system_prompt or ""
                modified_agent.system_prompt = f"{original_prompt} {style}"
                modified_agent.system_prompt_hash = ""
                modified_agent.model_post_init(None)

        elif ptype == "context_poisoning":
            injection = perturbation.get("injection_text", "")
            if injection:
                prompts = [f"{injection}{p}" for p in prompts]

        elif ptype == "tool_manipulation":
            pass

        return modified_agent, prompts, perturbation_record

    def _load_scenario(self, scenario_id: str):
        """Dynamically load a scenario module."""
        try:
            return importlib.import_module(f"scenarios.{scenario_id}")
        except ModuleNotFoundError:
            raise ValueError(f"Unknown scenario: {scenario_id}")

    @staticmethod
    def available_scenarios() -> list[str]:
        return [
            "healthy_baseline",
            "prompt_injection",
            "model_swap",
            "temperature_drift",
            "context_poisoning",
        ]
