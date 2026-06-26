import pytest

from domain.enums import AgentStatus


class TestRepositoryAgents:
    def test_save_and_get_agent(self, repository, sample_agent):
        repository.save_agent(sample_agent)
        retrieved = repository.get_agent(sample_agent.agent_id)
        assert retrieved is not None
        assert retrieved.agent_id == sample_agent.agent_id
        assert retrieved.display_name == sample_agent.display_name
        assert retrieved.model_id == sample_agent.model_id

    def test_get_nonexistent_agent(self, repository):
        assert repository.get_agent("nonexistent") is None

    def test_list_agents(self, repository, sample_agent, sample_agent_beta):
        repository.save_agent(sample_agent)
        repository.save_agent(sample_agent_beta)
        agents = repository.list_agents()
        assert len(agents) == 2

    def test_update_agent_status(self, repository, sample_agent):
        repository.save_agent(sample_agent)
        repository.update_agent_status(sample_agent.agent_id, AgentStatus.ACTIVE)
        retrieved = repository.get_agent(sample_agent.agent_id)
        assert retrieved.status == AgentStatus.ACTIVE


class TestRepositoryRuns:
    def test_save_and_get_runs(self, repository, sample_agent, sample_run):
        repository.save_agent(sample_agent)
        repository.save_run(sample_run)
        runs = repository.get_runs_for_agent(sample_agent.agent_id)
        assert len(runs) == 1
        assert runs[0].run_id == sample_run.run_id
        assert runs[0].prompt_text == sample_run.prompt_text

    def test_get_runs_empty(self, repository):
        runs = repository.get_runs_for_agent("nonexistent")
        assert runs == []


class TestRepositoryMetrics:
    def test_save_and_get_metrics(self, repository, sample_agent, sample_run, metric_extractor):
        repository.save_agent(sample_agent)
        repository.save_run(sample_run)
        metrics = metric_extractor.extract(sample_run)
        repository.save_metrics(metrics)
        retrieved = repository.get_metrics_for_agent(sample_agent.agent_id)
        assert len(retrieved) == 32


class TestRepositorySignatures:
    def test_save_and_get_baseline(self, repository, sample_agent, sample_signature):
        repository.save_agent(sample_agent)
        repository.save_signature(sample_signature)
        baseline = repository.get_baseline_signature(sample_agent.agent_id)
        assert baseline is not None
        assert baseline.signature_id == sample_signature.signature_id
        assert baseline.embedding_vector == sample_signature.embedding_vector

    def test_get_baseline_none(self, repository):
        assert repository.get_baseline_signature("nonexistent") is None


class TestRepositoryDrift:
    def test_save_drift_measurement(self, repository, sample_agent, sample_drift):
        repository.save_agent(sample_agent)
        repository.save_drift_measurement(sample_drift)


class TestRepositoryAudit:
    def test_log_audit_event(self, repository):
        repository.log_audit_event(
            source_component="test",
            event_type="test_event",
            agent_id="agent-1",
            payload={"key": "value"},
        )

    def test_audit_chain_integrity(self, repository):
        repository.log_audit_event("test", "event_1")
        repository.log_audit_event("test", "event_2")
