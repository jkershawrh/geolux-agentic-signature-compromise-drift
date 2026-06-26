import pytest

from domain.enums import (
    AgentStatus,
    CompromiseType,
    DriftCategory,
    MetricDimension,
    Reducibility,
    RubricState,
    RunStatus,
    SignatureType,
)
from domain.geometry import DriftMeasurement, GeometricSignature
from domain.lifecycle import is_valid_agent_transition, is_valid_run_transition
from domain.metrics import ALL_METRIC_NAMES, METRIC_DEFINITIONS, MetricMeasurement, get_exclusion_mask
from domain.models import AgentProfile, ControlledRun
from domain.reducibility import ReducibilityClassification


class TestAgentProfile:
    def test_create_with_defaults(self):
        agent = AgentProfile(
            display_name="Test Agent",
            model_id="claude-sonnet-4-20250514",
        )
        assert agent.display_name == "Test Agent"
        assert agent.model_id == "claude-sonnet-4-20250514"
        assert agent.status == AgentStatus.BASELINE_PENDING
        assert agent.agent_id  # auto-generated

    def test_system_prompt_hash_auto_computed(self):
        agent = AgentProfile(
            display_name="Test Agent",
            model_id="claude-sonnet-4-20250514",
            system_prompt="You are helpful.",
        )
        assert agent.system_prompt_hash != ""
        assert len(agent.system_prompt_hash) == 64

    def test_empty_display_name_raises(self):
        with pytest.raises(ValueError, match="display_name must not be empty"):
            AgentProfile(display_name="   ", model_id="claude-sonnet-4-20250514")

    def test_empty_model_id_raises(self):
        with pytest.raises(ValueError, match="model_id must not be empty"):
            AgentProfile(display_name="Test", model_id="  ")

    def test_display_name_stripped(self):
        agent = AgentProfile(display_name="  Test  ", model_id="claude-sonnet-4-20250514")
        assert agent.display_name == "Test"


class TestControlledRun:
    def test_create_minimal(self):
        run = ControlledRun(
            agent_id="agent-1",
            scenario_id="baseline",
            prompt_text="Hello",
            model_id="claude-sonnet-4-20250514",
        )
        assert run.status == RunStatus.PENDING
        assert run.prompt_hash != ""
        assert run.input_tokens == 0

    def test_prompt_hash_auto_computed(self):
        run = ControlledRun(
            agent_id="agent-1",
            scenario_id="baseline",
            prompt_text="Hello world",
            model_id="claude-sonnet-4-20250514",
        )
        assert len(run.prompt_hash) == 64

    def test_tool_sequence_auto_extracted(self):
        run = ControlledRun(
            agent_id="agent-1",
            scenario_id="baseline",
            prompt_text="test",
            model_id="claude-sonnet-4-20250514",
            tool_calls=[
                {"name": "search", "input": {}},
                {"name": "read", "input": {}},
            ],
        )
        assert run.tool_sequence == ["search", "read"]
        assert run.tool_call_count == 2

    def test_negative_tokens_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            ControlledRun(
                agent_id="agent-1",
                scenario_id="baseline",
                prompt_text="test",
                model_id="claude-sonnet-4-20250514",
                input_tokens=-1,
            )

    def test_empty_agent_id_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            ControlledRun(
                agent_id="   ",
                scenario_id="baseline",
                prompt_text="test",
                model_id="claude-sonnet-4-20250514",
            )


class TestMetricMeasurement:
    def test_create_valid(self):
        m = MetricMeasurement(
            run_id="run-1",
            agent_id="agent-1",
            dimension=MetricDimension.RESPONSE_STRUCTURE,
            metric_name="avg_response_length",
            value=150.0,
            normalized_value=0.3,
        )
        assert m.normalized_value == 0.3
        assert m.dimension == MetricDimension.RESPONSE_STRUCTURE

    def test_normalized_out_of_range_raises(self):
        with pytest.raises(ValueError, match="normalized_value must be in"):
            MetricMeasurement(
                run_id="run-1",
                agent_id="agent-1",
                dimension=MetricDimension.TOKEN_ECONOMICS,
                metric_name="input_output_ratio",
                value=1.5,
                normalized_value=1.5,
            )

    def test_negative_normalized_raises(self):
        with pytest.raises(ValueError, match="normalized_value must be in"):
            MetricMeasurement(
                run_id="run-1",
                agent_id="agent-1",
                dimension=MetricDimension.TOKEN_ECONOMICS,
                metric_name="input_output_ratio",
                value=0.5,
                normalized_value=-0.1,
            )

    def test_empty_metric_name_raises(self):
        with pytest.raises(ValueError, match="metric_name must not be empty"):
            MetricMeasurement(
                run_id="run-1",
                agent_id="agent-1",
                dimension=MetricDimension.TOKEN_ECONOMICS,
                metric_name="  ",
                value=0.5,
                normalized_value=0.5,
            )


class TestMetricDefinitions:
    def test_all_dimensions_have_metrics(self):
        for dim in MetricDimension:
            assert dim in METRIC_DEFINITIONS
            assert len(METRIC_DEFINITIONS[dim]) > 0

    def test_all_metric_names_unique(self):
        assert len(ALL_METRIC_NAMES) == len(set(ALL_METRIC_NAMES))

    def test_total_metric_count(self):
        assert len(ALL_METRIC_NAMES) == 32

    def test_exclusion_mask_length_matches_metrics(self):
        mask = get_exclusion_mask()
        assert len(mask) == 32

    def test_exclusion_mask_excludes_none(self):
        mask = get_exclusion_mask()
        assert mask.count(False) == 0
        assert all(mask)


class TestGeometricSignature:
    def test_create_valid(self, sample_signature):
        assert sample_signature.embedding_dimension == 7
        assert len(sample_signature.embedding_vector) == 7
        assert sample_signature.stability_score == 0.85

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="embedding_dimension"):
            GeometricSignature(
                agent_id="agent-1",
                signature_type=SignatureType.BASELINE,
                embedding_vector=[0.1, 0.2, 0.3],
                embedding_dimension=5,
                manifold_coordinates=[0.5],
                metric_snapshot={"a": 1.0},
                run_ids=["r1"],
                num_runs=1,
                computation_method="umap",
            )

    def test_empty_embedding_raises(self):
        with pytest.raises(ValueError, match="embedding_vector must not be empty"):
            GeometricSignature(
                agent_id="agent-1",
                signature_type=SignatureType.BASELINE,
                embedding_vector=[],
                embedding_dimension=0,
                manifold_coordinates=[],
                metric_snapshot={},
                run_ids=["r1"],
                num_runs=1,
                computation_method="umap",
            )

    def test_stability_out_of_range_raises(self):
        with pytest.raises(ValueError, match="stability_score must be in"):
            GeometricSignature(
                agent_id="agent-1",
                signature_type=SignatureType.SNAPSHOT,
                embedding_vector=[0.1],
                embedding_dimension=1,
                manifold_coordinates=[0.5],
                metric_snapshot={"a": 1.0},
                run_ids=["r1"],
                num_runs=1,
                computation_method="umap",
                stability_score=1.5,
            )

    def test_zero_num_runs_raises(self):
        with pytest.raises(ValueError, match="num_runs must be positive"):
            GeometricSignature(
                agent_id="agent-1",
                signature_type=SignatureType.BASELINE,
                embedding_vector=[0.1],
                embedding_dimension=1,
                manifold_coordinates=[0.5],
                metric_snapshot={"a": 1.0},
                run_ids=["r1"],
                num_runs=0,
                computation_method="umap",
            )


class TestDriftMeasurement:
    def test_create_valid(self, sample_drift):
        assert sample_drift.geodesic_distance == 0.45
        assert sample_drift.is_significant is True
        assert sample_drift.compromise_probability == 0.65

    def test_negative_distance_raises(self):
        with pytest.raises(ValueError, match="distance must be non-negative"):
            DriftMeasurement(
                agent_id="a",
                baseline_signature_id="s1",
                current_signature_id="s2",
                geodesic_distance=-0.1,
                euclidean_distance=0.1,
                cosine_similarity=0.9,
                drift_category=DriftCategory.GOAL,
                drift_magnitude=0.1,
                per_dimension_drift={},
                is_significant=False,
                compromise_probability=0.1,
            )

    def test_cosine_out_of_range_raises(self):
        with pytest.raises(ValueError, match="cosine_similarity must be in"):
            DriftMeasurement(
                agent_id="a",
                baseline_signature_id="s1",
                current_signature_id="s2",
                geodesic_distance=0.1,
                euclidean_distance=0.1,
                cosine_similarity=1.5,
                drift_category=DriftCategory.GOAL,
                drift_magnitude=0.1,
                per_dimension_drift={},
                is_significant=False,
                compromise_probability=0.1,
            )

    def test_compromise_probability_out_of_range_raises(self):
        with pytest.raises(ValueError, match="compromise_probability must be in"):
            DriftMeasurement(
                agent_id="a",
                baseline_signature_id="s1",
                current_signature_id="s2",
                geodesic_distance=0.1,
                euclidean_distance=0.1,
                cosine_similarity=0.9,
                drift_category=DriftCategory.GOAL,
                drift_magnitude=0.1,
                per_dimension_drift={},
                is_significant=False,
                compromise_probability=1.2,
            )


class TestReducibilityClassification:
    def test_create_valid(self):
        rc = ReducibilityClassification(
            agent_id="agent-1",
            dimension=MetricDimension.TOKEN_ECONOMICS,
            metric_name="input_output_ratio",
            reducibility=Reducibility.REDUCIBLE,
            predictability_score=0.95,
            variance=0.01,
            evidence={"test_statistic": 0.98},
            sample_size=30,
        )
        assert rc.reducibility == Reducibility.REDUCIBLE
        assert rc.predictability_score == 0.95

    def test_negative_variance_raises(self):
        with pytest.raises(ValueError, match="variance must be non-negative"):
            ReducibilityClassification(
                agent_id="agent-1",
                dimension=MetricDimension.TOKEN_ECONOMICS,
                metric_name="test",
                reducibility=Reducibility.REDUCIBLE,
                predictability_score=0.5,
                variance=-0.1,
                evidence={},
                sample_size=10,
            )

    def test_zero_sample_size_raises(self):
        with pytest.raises(ValueError, match="sample_size must be positive"):
            ReducibilityClassification(
                agent_id="agent-1",
                dimension=MetricDimension.TOKEN_ECONOMICS,
                metric_name="test",
                reducibility=Reducibility.REDUCIBLE,
                predictability_score=0.5,
                variance=0.1,
                evidence={},
                sample_size=0,
            )


class TestLifecycleTransitions:
    def test_valid_agent_transitions(self):
        assert is_valid_agent_transition(AgentStatus.BASELINE_PENDING, AgentStatus.ACTIVE)
        assert is_valid_agent_transition(AgentStatus.ACTIVE, AgentStatus.COMPROMISED)
        assert is_valid_agent_transition(AgentStatus.COMPROMISED, AgentStatus.RECOVERED)
        assert is_valid_agent_transition(AgentStatus.RECOVERED, AgentStatus.ACTIVE)
        assert is_valid_agent_transition(AgentStatus.ACTIVE, AgentStatus.ARCHIVED)

    def test_invalid_agent_transitions(self):
        assert not is_valid_agent_transition(AgentStatus.BASELINE_PENDING, AgentStatus.COMPROMISED)
        assert not is_valid_agent_transition(AgentStatus.ARCHIVED, AgentStatus.ACTIVE)
        assert not is_valid_agent_transition(AgentStatus.COMPROMISED, AgentStatus.ACTIVE)

    def test_valid_run_transitions(self):
        from domain.enums import RunStatus

        assert is_valid_run_transition(RunStatus.PENDING, RunStatus.RUNNING)
        assert is_valid_run_transition(RunStatus.RUNNING, RunStatus.COMPLETED)
        assert is_valid_run_transition(RunStatus.RUNNING, RunStatus.FAILED)

    def test_invalid_run_transitions(self):
        from domain.enums import RunStatus

        assert not is_valid_run_transition(RunStatus.PENDING, RunStatus.COMPLETED)
        assert not is_valid_run_transition(RunStatus.COMPLETED, RunStatus.RUNNING)
        assert not is_valid_run_transition(RunStatus.FAILED, RunStatus.RUNNING)


class TestEnums:
    def test_all_metric_dimensions(self):
        assert len(MetricDimension) == 8

    def test_all_drift_categories(self):
        assert len(DriftCategory) == 5

    def test_all_compromise_types(self):
        assert len(CompromiseType) == 5

    def test_all_rubric_states(self):
        assert len(RubricState) == 3

    def test_all_reducibility_values(self):
        assert len(Reducibility) == 3
