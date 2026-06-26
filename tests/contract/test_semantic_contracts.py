"""Contract tests verifying semantic analyzer output types and structure."""
from __future__ import annotations

import pytest

from adapters.mock_adapter import MockInferenceAdapter
from domain.semantics import SemanticDriftReport, SemanticSimilarityResult
from engine.semantic_analyzer import SemanticAnalyzer


@pytest.fixture
def analyzer():
    adapter = MockInferenceAdapter()
    return SemanticAnalyzer(adapter=adapter, judge_model_id="mock-judge")


class TestSemanticContracts:
    def test_output_is_semantic_similarity_result(self, analyzer):
        """compare_responses must return a SemanticSimilarityResult."""
        result = analyzer.compare_responses(
            prompt="Test prompt",
            baseline_response="Response A text",
            current_response="Response B text",
            structural_similarity=0.85,
            agent_id="contract-test",
        )
        assert isinstance(result, SemanticSimilarityResult)
        assert result.agent_id == "contract-test"
        assert result.judge_model_id == "mock-judge"
        assert 0.0 <= result.similarity_score <= 1.0
        assert 0.0 <= result.structural_similarity <= 1.0

    def test_report_is_semantic_drift_report(self, analyzer):
        """analyze_run_pair must return a SemanticDriftReport."""
        baseline_runs = [
            {"run_id": "b1", "prompt": "Q1", "response": "Answer one"},
            {"run_id": "b2", "prompt": "Q2", "response": "Answer two"},
        ]
        current_runs = [
            {"run_id": "c1", "prompt": "Q1", "response": "Answer one"},
            {"run_id": "c2", "prompt": "Q2", "response": "Answer two"},
        ]
        report = analyzer.analyze_run_pair(
            baseline_runs, current_runs, [0.9, 0.85], agent_id="contract-test",
        )
        assert isinstance(report, SemanticDriftReport)
        assert report.agent_id == "contract-test"
        assert 0.0 <= report.gaming_confidence <= 1.0

    def test_results_list_matches_input_pairs_count(self, analyzer):
        """Report results list must have the same count as input pairs."""
        n_pairs = 5
        baseline_runs = [
            {"run_id": f"b{i}", "prompt": f"Q{i}", "response": f"Base answer {i}"}
            for i in range(n_pairs)
        ]
        current_runs = [
            {"run_id": f"c{i}", "prompt": f"Q{i}", "response": f"Current answer {i}"}
            for i in range(n_pairs)
        ]
        structural_sims = [0.8 + i * 0.02 for i in range(n_pairs)]

        report = analyzer.analyze_run_pair(
            baseline_runs, current_runs, structural_sims, agent_id="contract-test",
        )
        assert len(report.results) == n_pairs
