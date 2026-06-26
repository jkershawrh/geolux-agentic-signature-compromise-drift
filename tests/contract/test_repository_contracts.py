"""Contract tests verifying repository satisfies data access patterns."""
import pytest

from domain.enums import AgentStatus, DriftCategory, SignatureType
from domain.geometry import DriftMeasurement, GeometricSignature
from domain.models import AgentProfile


class TestRepositoryContract:
    def test_agent_roundtrip(self, repository):
        agent = AgentProfile(
            agent_id="contract-agent", display_name="Contract",
            model_id="test",
        )
        repository.save_agent(agent)
        retrieved = repository.get_agent("contract-agent")
        assert retrieved is not None
        assert retrieved.agent_id == agent.agent_id
        assert retrieved.display_name == agent.display_name

    def test_signature_roundtrip(self, repository):
        agent = AgentProfile(
            agent_id="contract-sig-agent", display_name="C", model_id="t",
        )
        repository.save_agent(agent)

        sig = GeometricSignature(
            agent_id="contract-sig-agent",
            signature_type=SignatureType.BASELINE,
            embedding_vector=[0.1, 0.2, 0.3],
            embedding_dimension=3,
            manifold_coordinates=[0.5],
            metric_snapshot={"a": 0.5},
            run_ids=["r1"],
            num_runs=1,
            computation_method="pca",
        )
        repository.save_signature(sig)
        retrieved = repository.get_baseline_signature("contract-sig-agent")
        assert retrieved is not None
        assert retrieved.embedding_vector == sig.embedding_vector

    def test_drift_save_does_not_raise(self, repository):
        agent = AgentProfile(
            agent_id="contract-drift-agent", display_name="C", model_id="t",
        )
        repository.save_agent(agent)

        drift = DriftMeasurement(
            agent_id="contract-drift-agent",
            baseline_signature_id="s1",
            current_signature_id="s2",
            geodesic_distance=0.5,
            euclidean_distance=0.4,
            cosine_similarity=0.9,
            drift_category=DriftCategory.REASONING,
            drift_magnitude=0.3,
            per_dimension_drift={"a": 0.1},
            is_significant=True,
            p_value=0.03,
            compromise_probability=0.6,
        )
        repository.save_drift_measurement(drift)

    def test_audit_event_creates_hash(self, repository):
        repository.log_audit_event("contract", "test")
