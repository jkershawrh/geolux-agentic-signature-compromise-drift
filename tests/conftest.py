import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from db.models import Base
from db.repository import Repository
from domain.enums import (
    AgentStatus,
    DriftCategory,
    MetricDimension,
    Reducibility,
    SignatureType,
)
from domain.geometry import DriftMeasurement, GeometricSignature
from domain.metrics import MetricMeasurement
from domain.models import AgentProfile, ControlledRun
from domain.reducibility import ReducibilityClassification


@pytest.fixture
def db_engine():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    session_factory = sessionmaker(bind=db_engine)
    session = session_factory()
    yield session
    session.close()


@pytest.fixture
def repository(db_session):
    return Repository(db_session)


@pytest.fixture
def mock_adapter():
    return MockInferenceAdapter()


@pytest.fixture
def mock_adapter_with_tools():
    return MockInferenceAdapter(
        response_key="tool_use",
        include_tool_calls=True,
        latency_ms=250,
        input_tokens=200,
        output_tokens=100,
        thinking_tokens=50,
    )


@pytest.fixture
def metric_extractor():
    return DefaultMetricExtractor()


@pytest.fixture
def sample_agent():
    return AgentProfile(
        agent_id="test-agent-001",
        display_name="Test Agent Alpha",
        model_id="claude-sonnet-4-20250514",
        system_prompt="You are a helpful assistant.",
        configuration={"temperature": 1.0},
    )


@pytest.fixture
def sample_agent_beta():
    return AgentProfile(
        agent_id="test-agent-002",
        display_name="Test Agent Beta",
        model_id="claude-opus-4-20250514",
        system_prompt="You are a concise technical writer.",
        configuration={"temperature": 1.0},
    )


@pytest.fixture
def sample_run(sample_agent):
    return ControlledRun(
        run_id="run-001",
        agent_id=sample_agent.agent_id,
        scenario_id="healthy_baseline",
        prompt_text="What is the capital of France?",
        response_text="The capital of France is Paris. It is known for the Eiffel Tower.",
        model_id=sample_agent.model_id,
        input_tokens=100,
        output_tokens=50,
        latency_ms=200,
        status="completed",
    )


@pytest.fixture
def sample_run_with_tools(sample_agent):
    return ControlledRun(
        run_id="run-002",
        agent_id=sample_agent.agent_id,
        scenario_id="healthy_baseline",
        prompt_text="Find information about Python.",
        response_text="Here is a code example:\n\n```python\ndef hello():\n    print('Hello')\n```\n\nPython is great.",
        model_id=sample_agent.model_id,
        input_tokens=200,
        output_tokens=100,
        latency_ms=350,
        thinking_tokens=50,
        tool_calls=[
            {"name": "search", "input": {"query": "Python"}, "id": "tc_1"},
            {"name": "read_file", "input": {"path": "test.py"}, "id": "tc_2"},
        ],
        status="completed",
    )


@pytest.fixture
def sample_signature(sample_agent):
    return GeometricSignature(
        signature_id="sig-001",
        agent_id=sample_agent.agent_id,
        signature_type=SignatureType.BASELINE,
        embedding_vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        embedding_dimension=7,
        manifold_coordinates=[0.5, 0.3],
        metric_snapshot={"avg_response_length": 0.5, "input_output_ratio": 0.3},
        run_ids=["run-001", "run-002"],
        num_runs=2,
        computation_method="umap",
        stability_score=0.85,
    )


@pytest.fixture
def sample_drift(sample_agent):
    return DriftMeasurement(
        measurement_id="drift-001",
        agent_id=sample_agent.agent_id,
        baseline_signature_id="sig-001",
        current_signature_id="sig-002",
        geodesic_distance=0.45,
        euclidean_distance=0.38,
        cosine_similarity=0.82,
        drift_category=DriftCategory.REASONING,
        drift_magnitude=0.3,
        per_dimension_drift={"response_structure": 0.1, "reasoning_pattern": 0.5},
        is_significant=True,
        p_value=0.02,
        compromise_probability=0.65,
    )
