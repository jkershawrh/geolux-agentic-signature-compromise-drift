from __future__ import annotations

import pytest
from hypothesis import given as hgiven
from hypothesis import settings
from hypothesis import strategies as st

from adapters.mock_adapter import MockInferenceAdapter
from engine.semantic_analyzer import SemanticAnalyzer


@pytest.fixture
def analyzer():
    adapter = MockInferenceAdapter()
    return SemanticAnalyzer(adapter=adapter, judge_model_id="mock-judge")


class TestSemanticProperties:
    @hgiven(
        structural=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_semantic_gap_equals_structural_minus_semantic(self, structural):
        """semantic_gap must always equal structural_similarity - similarity_score."""
        adapter = MockInferenceAdapter()
        analyzer = SemanticAnalyzer(adapter=adapter, judge_model_id="mock-judge")
        result = analyzer.compare_responses(
            prompt="Property test prompt",
            baseline_response="Some baseline text here",
            current_response="Some current text here",
            structural_similarity=structural,
            agent_id="prop-test",
        )
        expected_gap = result.structural_similarity - result.similarity_score
        assert abs(result.semantic_gap - expected_gap) < 1e-6

    @hgiven(
        structural=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_gaming_confidence_in_range(self, structural):
        """gaming_confidence must always be in [0, 1]."""
        adapter = MockInferenceAdapter()
        analyzer = SemanticAnalyzer(adapter=adapter, judge_model_id="mock-judge")
        baseline_runs = [
            {"run_id": "b1", "prompt": "Q", "response": "Alpha beta gamma"},
        ]
        current_runs = [
            {"run_id": "c1", "prompt": "Q", "response": "Delta epsilon zeta"},
        ]
        report = analyzer.analyze_run_pair(
            baseline_runs, current_runs, [structural], agent_id="prop-test",
        )
        assert 0.0 <= report.gaming_confidence <= 1.0

    def test_identical_text_always_high_similarity(self):
        """Identical text should always produce similarity >= 0.8."""
        adapter = MockInferenceAdapter()
        analyzer = SemanticAnalyzer(adapter=adapter, judge_model_id="mock-judge")
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a popular programming language.",
            "Machine learning and artificial intelligence are transforming industries.",
            "The capital of France is Paris.",
            "Water boils at one hundred degrees Celsius.",
        ]
        for text in texts:
            result = analyzer.compare_responses(
                prompt="Test",
                baseline_response=text,
                current_response=text,
                structural_similarity=0.95,
                agent_id="prop-test",
            )
            assert result.similarity_score >= 0.8, (
                f"Identical text '{text[:30]}...' got similarity "
                f"{result.similarity_score}"
            )
