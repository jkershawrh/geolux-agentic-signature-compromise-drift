
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import RunStatus


class TestMockInferenceAdapter:
    def test_execute_returns_completed_run(self, mock_adapter, sample_agent):
        run = mock_adapter.execute(sample_agent, "Hello")
        assert run.status == RunStatus.COMPLETED
        assert run.agent_id == sample_agent.agent_id
        assert run.model_id == sample_agent.model_id

    def test_execute_sets_prompt_hash(self, mock_adapter, sample_agent):
        run = mock_adapter.execute(sample_agent, "Hello")
        assert len(run.prompt_hash) == 64

    def test_execute_with_tools(self, mock_adapter_with_tools, sample_agent):
        run = mock_adapter_with_tools.execute(sample_agent, "test")
        assert run.tool_call_count == 2
        assert run.tool_sequence == ["search", "read_file"]
        assert run.thinking_tokens == 50

    def test_execute_without_tools(self, mock_adapter, sample_agent):
        run = mock_adapter.execute(sample_agent, "test")
        assert run.tool_call_count == 0
        assert run.tool_sequence == []

    def test_deterministic_response(self, mock_adapter, sample_agent):
        run1 = mock_adapter.execute(sample_agent, "test")
        run2 = mock_adapter.execute(sample_agent, "test")
        assert run1.response_text == run2.response_text

    def test_different_response_keys(self, sample_agent):
        adapter = MockInferenceAdapter(response_key="reasoning")
        run = adapter.execute(sample_agent, "test")
        assert "step by step" in run.response_text

    def test_refusal_response(self, sample_agent):
        adapter = MockInferenceAdapter(response_key="refusal", stop_reason="refusal")
        run = adapter.execute(sample_agent, "test")
        assert run.stop_reason == "refusal"
        assert "not able" in run.response_text
