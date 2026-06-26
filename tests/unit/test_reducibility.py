import numpy as np
import pytest

from adapters.metric_extractor import DefaultMetricExtractor
from adapters.mock_adapter import MockInferenceAdapter
from domain.enums import Reducibility
from domain.models import AgentProfile
from engine.reducibility_analyzer import ReducibilityAnalyzer


def _collect_metrics(adapter, n=10):
    extractor = DefaultMetricExtractor()
    agent = AgentProfile(agent_id="test", display_name="Test", model_id="test")
    return [extractor.extract(adapter.execute(agent, f"Prompt {i}")) for i in range(n)]


class TestReducibilityAnalyzer:
    def test_analyze_produces_32_classifications(self):
        adapter = MockInferenceAdapter()
        metrics_list = _collect_metrics(adapter)
        analyzer = ReducibilityAnalyzer(min_samples=5)
        classifications = analyzer.analyze(metrics_list, "test")
        assert len(classifications) == 32

    def test_too_few_samples_raises(self):
        adapter = MockInferenceAdapter()
        metrics_list = _collect_metrics(adapter, n=2)
        analyzer = ReducibilityAnalyzer(min_samples=5)
        with pytest.raises(ValueError, match="at least 5"):
            analyzer.analyze(metrics_list, "test")

    def test_deterministic_metrics_are_reducible(self):
        adapter = MockInferenceAdapter()
        metrics_list = _collect_metrics(adapter)
        analyzer = ReducibilityAnalyzer(
            reducible_variance_threshold=0.1,
            min_samples=5,
        )
        classifications = analyzer.analyze(metrics_list, "test")
        reducible = [c for c in classifications if c.reducibility == Reducibility.REDUCIBLE]
        assert len(reducible) > 0

    def test_predictability_in_range(self):
        adapter = MockInferenceAdapter()
        metrics_list = _collect_metrics(adapter)
        analyzer = ReducibilityAnalyzer(min_samples=5)
        classifications = analyzer.analyze(metrics_list, "test")
        for c in classifications:
            assert 0.0 <= c.predictability_score <= 1.0

    def test_variance_non_negative(self):
        adapter = MockInferenceAdapter()
        metrics_list = _collect_metrics(adapter)
        analyzer = ReducibilityAnalyzer(min_samples=5)
        classifications = analyzer.analyze(metrics_list, "test")
        for c in classifications:
            assert c.variance >= 0

    def test_evidence_populated(self):
        adapter = MockInferenceAdapter()
        metrics_list = _collect_metrics(adapter)
        analyzer = ReducibilityAnalyzer(min_samples=5)
        classifications = analyzer.analyze(metrics_list, "test")
        for c in classifications:
            assert "variance" in c.evidence
            assert "mean" in c.evidence
            assert "autocorrelation_lag1" in c.evidence

    def test_reducible_mask(self):
        adapter = MockInferenceAdapter()
        metrics_list = _collect_metrics(adapter)
        analyzer = ReducibilityAnalyzer(min_samples=5)
        classifications = analyzer.analyze(metrics_list, "test")
        mask = analyzer.get_reducible_mask(classifications)
        assert len(mask) == 32
        assert all(isinstance(m, bool) for m in mask)

    def test_summary_counts(self):
        adapter = MockInferenceAdapter()
        metrics_list = _collect_metrics(adapter)
        analyzer = ReducibilityAnalyzer(min_samples=5)
        classifications = analyzer.analyze(metrics_list, "test")
        summary = analyzer.summary(classifications)
        total = sum(summary.values())
        assert total == 32

    def test_compute_fisher_ratios_returns_all_metrics(self):
        """Fisher ratios should have an entry for each of 32 metrics."""
        analyzer = ReducibilityAnalyzer()
        a = np.random.RandomState(42).rand(10, 32)
        b = np.random.RandomState(43).rand(10, 32)
        ratios = analyzer.compute_fisher_ratios(a, b)
        assert len(ratios) == 32
        assert all(v >= 0 for v in ratios.values())

    def test_fisher_ratio_high_for_separated_metrics(self):
        """Metrics with large mean difference and low variance should have high Fisher ratio."""
        analyzer = ReducibilityAnalyzer()
        a = np.zeros((10, 32))
        b = np.zeros((10, 32))
        a[:, 0] = 0.1  # metric 0: stable at 0.1 for agent A
        b[:, 0] = 0.9  # metric 0: stable at 0.9 for agent B
        a[:, 1] = np.random.RandomState(42).rand(10)  # metric 1: random noise
        b[:, 1] = np.random.RandomState(43).rand(10)
        ratios = analyzer.compute_fisher_ratios(a, b)
        from domain.metrics import ALL_METRIC_NAMES
        assert ratios[ALL_METRIC_NAMES[0]] > ratios[ALL_METRIC_NAMES[1]]

    def test_discriminative_mask_selects_top_k(self):
        """Mask should have exactly top_k True values."""
        analyzer = ReducibilityAnalyzer()
        ratios = {f"metric_{i}": float(i) for i in range(32)}
        # Map to actual metric names
        from domain.metrics import ALL_METRIC_NAMES
        ratios = {name: float(i) for i, name in enumerate(ALL_METRIC_NAMES)}
        mask = analyzer.get_discriminative_mask(ratios, top_k=10)
        assert sum(mask) == 10
