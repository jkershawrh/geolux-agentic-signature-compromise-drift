from __future__ import annotations

import json

import pytest

from adapters.mock_adapter import MockInferenceAdapter
from domain.semantics import SemanticDriftReport, SemanticSimilarityResult
from engine.semantic_analyzer import SemanticAnalyzer


@pytest.fixture
def mock_adapter():
    return MockInferenceAdapter()


@pytest.fixture
def analyzer(mock_adapter):
    return SemanticAnalyzer(adapter=mock_adapter, judge_model_id="mock-judge")


class TestSemanticAnalyzer:
    def test_compare_identical_responses_high_similarity(self, analyzer):
        """Identical responses should produce high similarity via word overlap fallback."""
        text = "The capital of France is Paris."
        result = analyzer.compare_responses(
            prompt="What is the capital of France?",
            baseline_response=text,
            current_response=text,
            structural_similarity=0.95,
            agent_id="test-agent",
        )
        assert isinstance(result, SemanticSimilarityResult)
        # Word overlap of identical text = 1.0
        assert result.similarity_score >= 0.8

    def test_compare_different_responses_low_similarity(self, analyzer):
        """Completely different responses should produce low similarity."""
        result = analyzer.compare_responses(
            prompt="Tell me about science.",
            baseline_response="Quantum physics explores subatomic particles and wave functions.",
            current_response="Chocolate cake requires flour sugar eggs and butter to bake properly.",
            structural_similarity=0.9,
            agent_id="test-agent",
        )
        assert isinstance(result, SemanticSimilarityResult)
        # Very different word sets => low overlap
        assert result.similarity_score < 0.5

    def test_detect_gaming_when_structural_high_semantic_low(self, analyzer):
        """Gaming: high structural similarity but low semantic similarity."""
        result = analyzer.compare_responses(
            prompt="Explain gravity.",
            baseline_response="Quantum physics explores subatomic particles and wave functions.",
            current_response="Chocolate cake requires flour sugar eggs and butter to bake properly.",
            structural_similarity=0.95,
            agent_id="test-agent",
        )
        # structural (0.95) - semantic (low) => positive gap
        assert result.semantic_gap > 0.0

    def test_no_gaming_when_both_high(self, analyzer):
        """No gaming when structural and semantic are both high."""
        text = "The capital of France is Paris."
        result = analyzer.compare_responses(
            prompt="What is the capital of France?",
            baseline_response=text,
            current_response=text,
            structural_similarity=0.95,
            agent_id="test-agent",
        )
        # Identical => similarity ~1.0, structural 0.95, gap ~negative
        assert result.semantic_gap <= 0.1

    def test_judge_prompt_contains_both_responses(self, analyzer):
        """The judge prompt must include both responses and the question."""
        prompt_text = "What is Python?"
        response_a = "Python is a programming language."
        response_b = "Python is a snake."
        judge_prompt = analyzer._build_judge_prompt(
            prompt_text, response_a, response_b,
        )
        assert prompt_text in judge_prompt
        assert response_a in judge_prompt
        assert response_b in judge_prompt
        assert "semantic similarity" in judge_prompt.lower()

    def test_parse_judge_response_valid_json(self, analyzer):
        """Valid JSON responses are parsed correctly."""
        response = json.dumps({"score": 7, "reasoning": "Similar meaning"})
        score, reasoning = analyzer._parse_judge_response(response)
        assert score == pytest.approx(0.7, abs=0.01)
        assert reasoning == "Similar meaning"

    def test_parse_judge_response_fallback_on_invalid(self, analyzer):
        """Invalid JSON falls back to regex or returns 0."""
        response = "This is not JSON at all, just text about score: 8 for similarity."
        score, reasoning = analyzer._parse_judge_response(response)
        # Regex fallback should find "score: 8"
        assert score == pytest.approx(0.8, abs=0.01)

    def test_analyze_run_pair_aggregates_correctly(self, analyzer):
        """analyze_run_pair should aggregate results from multiple pairs."""
        baseline_runs = [
            {"run_id": "b1", "prompt": "Q1", "response": "Answer about topic A"},
            {"run_id": "b2", "prompt": "Q2", "response": "Answer about topic B"},
            {"run_id": "b3", "prompt": "Q3", "response": "Answer about topic C"},
        ]
        current_runs = [
            {"run_id": "c1", "prompt": "Q1", "response": "Answer about topic A"},
            {"run_id": "c2", "prompt": "Q2", "response": "Answer about topic B"},
            {"run_id": "c3", "prompt": "Q3", "response": "Answer about topic C"},
        ]
        structural_sims = [0.9, 0.85, 0.88]

        report = analyzer.analyze_run_pair(
            baseline_runs, current_runs, structural_sims, agent_id="test",
        )
        assert isinstance(report, SemanticDriftReport)
        assert len(report.results) == 3
        assert report.agent_id == "test"
        # Mean structural should match average of inputs
        expected_mean_struct = sum(structural_sims) / len(structural_sims)
        assert report.mean_structural_similarity == pytest.approx(
            expected_mean_struct, abs=0.01,
        )

    def test_semantic_gap_computed_correctly(self, analyzer):
        """semantic_gap should equal structural_similarity - similarity_score."""
        result = analyzer.compare_responses(
            prompt="Test prompt",
            baseline_response="The quick brown fox jumps over the lazy dog",
            current_response="The quick brown fox jumps over the lazy dog",
            structural_similarity=0.8,
            agent_id="test-agent",
        )
        expected_gap = result.structural_similarity - result.similarity_score
        assert result.semantic_gap == pytest.approx(expected_gap, abs=1e-6)

    def test_detect_gaming_factual_inversion(self):
        """Structurally similar but factually wrong responses should produce positive gap."""
        analyzer = SemanticAnalyzer(adapter=MockInferenceAdapter(), judge_model_id="test")
        result = analyzer.compare_responses(
            prompt="What is the capital of France?",
            baseline_response="The capital of France is Paris.",
            current_response="The capital of France is London.",
            structural_similarity=0.95,
        )
        # With stopword filtering, word overlap should be lower for these
        assert result.semantic_gap > 0.1

    def test_gaming_confidence_in_range(self, analyzer):
        """gaming_confidence must always be in [0, 1]."""
        baseline_runs = [
            {"run_id": "b1", "prompt": "Q1", "response": "Alpha beta gamma"},
        ]
        current_runs = [
            {"run_id": "c1", "prompt": "Q1", "response": "Delta epsilon zeta"},
        ]
        report = analyzer.analyze_run_pair(
            baseline_runs, current_runs, [0.95], agent_id="test",
        )
        assert 0.0 <= report.gaming_confidence <= 1.0
