"""Contract tests verifying canary system satisfies expected interfaces."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.canaries import CanaryReport, CanaryResult, CanaryType
from domain.models import AgentProfile
from engine.canary_system import CanarySystem


def _make_report() -> CanaryReport:
    system = CanarySystem()
    agent = AgentProfile(
        agent_id="contract-test-agent",
        display_name="Contract Test Agent",
        model_id="test-model",
    )
    adapter = MockInferenceAdapter()
    extractor = DefaultMetricExtractor()
    return system.execute_and_verify(agent, adapter, extractor)


class TestCanarySystemContract:
    def test_output_is_canary_report(self):
        report = _make_report()
        assert isinstance(report, CanaryReport)

    def test_report_results_contain_canary_result_instances(self):
        report = _make_report()
        assert len(report.results) > 0
        for r in report.results:
            assert isinstance(r, CanaryResult)

    def test_per_type_pass_rate_has_entry_for_each_active_type(self):
        report = _make_report()
        active_types = {ct for ct in CanaryType if ct != CanaryType.BEHAVIORAL_MULTI_TURN}
        for ctype in active_types:
            assert ctype.value in report.per_type_pass_rate, (
                f"Missing per_type_pass_rate entry for {ctype.value}"
            )
