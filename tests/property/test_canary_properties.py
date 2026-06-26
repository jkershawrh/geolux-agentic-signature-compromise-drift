from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adapters.mock_adapter import MockInferenceAdapter
from adapters.metric_extractor import DefaultMetricExtractor
from domain.canaries import CanaryReport, CanaryResult, CanaryType
from domain.models import AgentProfile
from engine.canary_system import CanarySystem


def _run_canary_report(adapter_key: str = "default") -> CanaryReport:
    system = CanarySystem()
    agent = AgentProfile(
        agent_id="prop-test-agent",
        display_name="Property Test Agent",
        model_id="test-model",
    )
    adapter = MockInferenceAdapter(response_key=adapter_key)
    extractor = DefaultMetricExtractor()
    return system.execute_and_verify(agent, adapter, extractor)


class TestCanaryProperties:
    @given(adapter_key=st.sampled_from(["default", "code", "reasoning", "refusal"]))
    @settings(max_examples=10)
    def test_pass_rate_always_in_0_1(self, adapter_key):
        report = _run_canary_report(adapter_key)
        assert 0.0 <= report.pass_rate <= 1.0

    @given(adapter_key=st.sampled_from(["default", "code", "reasoning", "refusal"]))
    @settings(max_examples=10)
    def test_per_type_pass_rate_keys_match_canary_types(self, adapter_key):
        report = _run_canary_report(adapter_key)
        # All types with default probes should appear (BEHAVIORAL_MULTI_TURN has none)
        active_types = {ct for ct in CanaryType if ct != CanaryType.BEHAVIORAL_MULTI_TURN}
        for ctype in active_types:
            assert ctype.value in report.per_type_pass_rate, (
                f"Missing key for {ctype.value}"
            )
        # Each value should be in [0, 1]
        for key, rate in report.per_type_pass_rate.items():
            assert 0.0 <= rate <= 1.0, f"Rate for {key} out of range: {rate}"

    @given(adapter_key=st.sampled_from(["default", "code", "reasoning", "refusal"]))
    @settings(max_examples=10)
    def test_authenticity_score_in_0_1(self, adapter_key):
        report = _run_canary_report(adapter_key)
        assert 0.0 <= report.authenticity_score <= 1.0
