from __future__ import annotations

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

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


def _build_simulator() -> tuple[AttackSimulator, RealisticMockAdapter]:
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
    return simulator, adapter


def _build_agent() -> AgentProfile:
    return AgentProfile(
        agent_id="prop-attack-agent",
        display_name="Property Attack Agent",
        model_id="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant.",
    )


def _build_baseline(vec: list[float]) -> GeometricSignature:
    return GeometricSignature(
        agent_id="prop-attack-agent",
        signature_type=SignatureType.BASELINE,
        embedding_vector=vec,
        embedding_dimension=len(vec),
        manifold_coordinates=[0.0, 0.0],
        metric_snapshot={"avg_response_length": 0.5},
        run_ids=["r1", "r2"],
        num_runs=2,
        computation_method="test",
        stability_score=0.85,
    )


class TestAttackProperties:
    @given(
        detection_rate=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_detection_plus_evasion_equals_one(self, detection_rate: float):
        evasion_rate = 1.0 - detection_rate
        result = AttackResult(
            attack_id="test",
            attack_type=AttackType.METRIC_GAMING,
            target_agent_id="agent",
            detection_rate=detection_rate,
            evasion_rate=evasion_rate,
            num_trials=1,
            detections=[],
            summary="test",
        )
        assert abs(result.detection_rate + result.evasion_rate - 1.0) < 1e-9

    @given(
        detection_rate=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_detection_rate_in_range(self, detection_rate: float):
        result = AttackResult(
            attack_id="test",
            attack_type=AttackType.PROMPT_MIMICRY,
            target_agent_id="agent",
            detection_rate=detection_rate,
            evasion_rate=1.0 - detection_rate,
            num_trials=1,
            detections=[],
            summary="test",
        )
        assert 0.0 <= result.detection_rate <= 1.0

    @given(
        num_trials=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_num_trials_positive(self, num_trials: int):
        result = AttackResult(
            attack_id="test",
            attack_type=AttackType.GRADUAL_DRIFT,
            target_agent_id="agent",
            detection_rate=0.5,
            evasion_rate=0.5,
            num_trials=num_trials,
            detections=[],
            summary="test",
        )
        assert result.num_trials > 0

    @given(
        num_detections=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=50)
    def test_detections_list_length_bounded_by_trials(self, num_detections: int):
        num_trials = max(num_detections, 1)
        detections = [{"trial": i, "detected": True} for i in range(num_detections)]
        result = AttackResult(
            attack_id="test",
            attack_type=AttackType.SIGNATURE_SPOOFING,
            target_agent_id="agent",
            detection_rate=num_detections / num_trials if num_trials > 0 else 0.0,
            evasion_rate=1.0 - (num_detections / num_trials if num_trials > 0 else 0.0),
            num_trials=num_trials,
            detections=detections,
            summary="test",
        )
        assert len(result.detections) <= result.num_trials
