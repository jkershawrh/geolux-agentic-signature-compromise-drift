from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockConversationalAdapter, RealisticMockAdapter
from domain.attacks import AttackResult, AttackType
from domain.enums import SignatureType
from domain.geometry import GeometricSignature
from domain.models import AgentProfile
from engine.attack_simulator import AttackSimulator
from engine.canary_system import CanarySystem
from engine.drift_detector import DriftDetector
from engine.semantic_analyzer import SemanticAnalyzer
from engine.signature_generator import SignatureGenerator
from engine.temporal_tracker import TemporalTracker


@pytest.fixture
def extractor():
    return DefaultMetricExtractor()


@pytest.fixture
def generator():
    return SignatureGenerator(min_runs=5)


@pytest.fixture
def drift_detector():
    return DriftDetector()


@pytest.fixture
def adapter():
    return RealisticMockAdapter(profile="coder")


@pytest.fixture
def semantic_analyzer(adapter):
    return SemanticAnalyzer(adapter=adapter)


@pytest.fixture
def canary_system():
    return CanarySystem()


@pytest.fixture
def temporal_tracker():
    return TemporalTracker(window_size=3)


@pytest.fixture
def agent():
    return AgentProfile(
        agent_id="attack-test-agent",
        display_name="Attack Test Agent",
        model_id="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant.",
    )


@pytest.fixture
def target_baseline():
    return GeometricSignature(
        agent_id="attack-test-agent",
        signature_type=SignatureType.BASELINE,
        embedding_vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        embedding_dimension=7,
        manifold_coordinates=[0.5, 0.3],
        metric_snapshot={"avg_response_length": 0.5, "input_output_ratio": 0.3},
        run_ids=["run-001", "run-002"],
        num_runs=2,
        computation_method="test",
        stability_score=0.85,
    )


@pytest.fixture
def simulator(extractor, generator, drift_detector, semantic_analyzer,
              canary_system, temporal_tracker):
    return AttackSimulator(
        extractor=extractor,
        generator=generator,
        drift_detector=drift_detector,
        semantic_analyzer=semantic_analyzer,
        canary_system=canary_system,
        temporal_tracker=temporal_tracker,
    )


class TestMetricGaming:
    def test_simulate_metric_gaming_returns_result(
        self, simulator, target_baseline, adapter, agent,
    ):
        result = simulator.simulate_metric_gaming(
            target_baseline, adapter, agent, num_trials=3,
        )
        assert isinstance(result, AttackResult)
        assert result.attack_type == AttackType.METRIC_GAMING
        assert result.target_agent_id == agent.agent_id

    def test_metric_gaming_detection_rate_in_range(
        self, simulator, target_baseline, adapter, agent,
    ):
        result = simulator.simulate_metric_gaming(
            target_baseline, adapter, agent, num_trials=3,
        )
        assert 0.0 <= result.detection_rate <= 1.0


class TestPromptMimicry:
    def test_simulate_prompt_mimicry_returns_result(
        self, simulator, target_baseline, adapter, agent,
    ):
        result = simulator.simulate_prompt_mimicry(
            target_baseline, adapter, agent, num_trials=3,
        )
        assert isinstance(result, AttackResult)
        assert result.attack_type == AttackType.PROMPT_MIMICRY
        assert result.target_agent_id == agent.agent_id

    def test_prompt_mimicry_detection_rate_in_range(
        self, simulator, target_baseline, adapter, agent,
    ):
        result = simulator.simulate_prompt_mimicry(
            target_baseline, adapter, agent, num_trials=3,
        )
        assert 0.0 <= result.detection_rate <= 1.0

    def test_prompt_mimicry_detects_compromised_mock(
        self, simulator, target_baseline, agent,
    ):
        """With mock adapter (compromised=True vs False), detection should be >0."""
        # Use a non-multi-turn adapter so the simulator uses its own
        # MockConversationalAdapter(compromised=True) internally.
        adapter = RealisticMockAdapter(profile="coder")
        result = simulator.simulate_prompt_mimicry(
            target_baseline, adapter, agent, num_trials=5,
        )
        assert result.detection_rate > 0, (
            f"Expected detection_rate > 0 but got {result.detection_rate}; "
            f"detections={result.detections}"
        )


class TestGradualDrift:
    def test_simulate_gradual_drift_returns_result(
        self, simulator, target_baseline, adapter, agent,
    ):
        result = simulator.simulate_gradual_drift(
            target_baseline, adapter, agent, num_trials=5,
        )
        assert isinstance(result, AttackResult)
        assert result.attack_type == AttackType.GRADUAL_DRIFT
        assert result.target_agent_id == agent.agent_id

    def test_gradual_drift_detection_rate_in_range(
        self, simulator, target_baseline, adapter, agent,
    ):
        result = simulator.simulate_gradual_drift(
            target_baseline, adapter, agent, num_trials=5,
        )
        assert 0.0 <= result.detection_rate <= 1.0


class TestSignatureSpoofing:
    def test_simulate_signature_spoofing_returns_result(
        self, simulator, target_baseline, adapter, agent,
    ):
        result = simulator.simulate_signature_spoofing(
            target_baseline, adapter, agent, num_trials=3,
        )
        assert isinstance(result, AttackResult)
        assert result.attack_type == AttackType.SIGNATURE_SPOOFING
        assert result.target_agent_id == agent.agent_id

    def test_signature_spoofing_detection_rate_in_range(
        self, simulator, target_baseline, adapter, agent,
    ):
        result = simulator.simulate_signature_spoofing(
            target_baseline, adapter, agent, num_trials=3,
        )
        assert 0.0 <= result.detection_rate <= 1.0


class TestAggregateAPI:
    def test_run_all_attacks_returns_four_results(
        self, simulator, target_baseline, adapter, agent,
    ):
        results = simulator.run_all_attacks(target_baseline, adapter, agent)
        assert len(results) == 4
        attack_types = {r.attack_type for r in results}
        assert attack_types == {
            AttackType.METRIC_GAMING,
            AttackType.PROMPT_MIMICRY,
            AttackType.GRADUAL_DRIFT,
            AttackType.SIGNATURE_SPOOFING,
        }

    def test_summary_report_has_all_attack_types(
        self, simulator, target_baseline, adapter, agent,
    ):
        results = simulator.run_all_attacks(target_baseline, adapter, agent)
        report = simulator.summary_report(results)
        assert "overall_detection_rate" in report
        assert "overall_evasion_rate" in report
        assert "total_trials" in report
        assert "per_attack" in report
        for atype in AttackType:
            assert atype.value in report["per_attack"]

    def test_evasion_rate_complements_detection_rate(
        self, simulator, target_baseline, adapter, agent,
    ):
        results = simulator.run_all_attacks(target_baseline, adapter, agent)
        for result in results:
            assert abs(result.detection_rate + result.evasion_rate - 1.0) < 1e-9

    def test_num_trials_matches_config(
        self, simulator, target_baseline, adapter, agent,
    ):
        result = simulator.simulate_metric_gaming(
            target_baseline, adapter, agent, num_trials=7,
        )
        assert result.num_trials == 7
        assert len(result.detections) == 7
