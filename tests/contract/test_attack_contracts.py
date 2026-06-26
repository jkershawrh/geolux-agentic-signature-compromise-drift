from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import RealisticMockAdapter
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


def _build_simulator_and_run() -> list[AttackResult]:
    extractor = DefaultMetricExtractor()
    generator = SignatureGenerator(min_runs=5)
    drift_detector = DriftDetector()
    adapter = RealisticMockAdapter(profile="coder")
    semantic_analyzer = SemanticAnalyzer(adapter=adapter)
    canary_system = CanarySystem()
    temporal_tracker = TemporalTracker(window_size=3)

    simulator = AttackSimulator(
        extractor=extractor,
        generator=generator,
        drift_detector=drift_detector,
        semantic_analyzer=semantic_analyzer,
        canary_system=canary_system,
        temporal_tracker=temporal_tracker,
    )

    agent = AgentProfile(
        agent_id="contract-attack-agent",
        display_name="Contract Attack Agent",
        model_id="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant.",
    )

    baseline = GeometricSignature(
        agent_id="contract-attack-agent",
        signature_type=SignatureType.BASELINE,
        embedding_vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        embedding_dimension=7,
        manifold_coordinates=[0.5, 0.3],
        metric_snapshot={"avg_response_length": 0.5},
        run_ids=["r1", "r2"],
        num_runs=2,
        computation_method="test",
        stability_score=0.85,
    )

    return simulator.run_all_attacks(baseline, adapter, agent)


class TestAttackContracts:
    def test_output_is_attack_result(self):
        results = _build_simulator_and_run()
        for result in results:
            assert isinstance(result, AttackResult)

    def test_attack_type_is_valid(self):
        results = _build_simulator_and_run()
        valid_types = set(AttackType)
        for result in results:
            assert result.attack_type in valid_types

    def test_run_all_attacks_returns_list_of_attack_result(self):
        results = _build_simulator_and_run()
        assert isinstance(results, list)
        assert len(results) == 4
        for result in results:
            assert isinstance(result, AttackResult)
