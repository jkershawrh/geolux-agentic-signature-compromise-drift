import numpy as np
import pytest

from domain.enums import MetricDimension
from domain.metrics import MetricMeasurement
from engine.geometric.embedding import (
    MetricVectorBuilder,
    aggregate_metric_vectors,
    metrics_to_vector,
    normalize_vector,
    pca_project,
    project_point_pca,
)


def _make_metric(dim, name, value, normalized):
    return MetricMeasurement(
        run_id="r1", agent_id="a1", dimension=dim,
        metric_name=name, value=value, normalized_value=normalized,
    )


class TestMetricsToVector:
    def test_produces_correct_length(self, metric_extractor, sample_run):
        metrics = metric_extractor.extract(sample_run)
        vec = metrics_to_vector(metrics)
        assert len(vec) == 32

    def test_values_from_normalized(self, metric_extractor, sample_run):
        metrics = metric_extractor.extract(sample_run)
        vec = metrics_to_vector(metrics)
        for v in vec:
            assert 0.0 <= v <= 1.0

    def test_missing_metric_defaults_to_zero(self):
        metrics = [
            _make_metric(MetricDimension.RESPONSE_STRUCTURE, "avg_response_length", 100, 0.5),
        ]
        vec = metrics_to_vector(metrics)
        assert vec[0] == 0.5
        assert vec[1] == 0.0  # missing metric defaults to 0


class TestAggregateVectors:
    def test_mean_of_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        result = aggregate_metric_vectors([v, v, v])
        np.testing.assert_array_almost_equal(result, v)

    def test_mean_of_two(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 1.0])
        result = aggregate_metric_vectors([v1, v2])
        np.testing.assert_array_almost_equal(result, [0.5, 0.5])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            aggregate_metric_vectors([])


class TestNormalizeVector:
    def test_with_bounds(self):
        vec = np.array([5.0, 10.0])
        result = normalize_vector(vec, np.array([0.0, 0.0]), np.array([10.0, 20.0]))
        np.testing.assert_array_almost_equal(result, [0.5, 0.5])

    def test_without_bounds_returns_same(self):
        vec = np.array([0.3, 0.7])
        result = normalize_vector(vec)
        np.testing.assert_array_almost_equal(result, vec)

    def test_zero_range_handled(self):
        vec = np.array([5.0, 5.0])
        result = normalize_vector(vec, np.array([5.0, 0.0]), np.array([5.0, 10.0]))
        assert result[0] == 0.0
        assert result[1] == 0.5


class TestMetricVectorBuilder:
    def test_build_and_centroid(self, metric_extractor, sample_run):
        builder = MetricVectorBuilder()
        metrics = metric_extractor.extract(sample_run)
        builder.add_metrics(metrics)
        builder.add_metrics(metrics)

        centroid = builder.get_centroid()
        assert len(centroid) == 32
        assert builder.sample_count == 2

    def test_covariance_needs_two(self, metric_extractor, sample_run):
        builder = MetricVectorBuilder()
        builder.add_metrics(metric_extractor.extract(sample_run))
        with pytest.raises(ValueError, match="at least 2"):
            builder.get_covariance()

    def test_covariance_shape(self, metric_extractor, sample_run):
        builder = MetricVectorBuilder()
        for _ in range(5):
            builder.add_metrics(metric_extractor.extract(sample_run))
        cov = builder.get_covariance()
        assert cov.shape == (32, 32)

    def test_history_bounds(self, metric_extractor, sample_run):
        builder = MetricVectorBuilder()
        builder.add_metrics(metric_extractor.extract(sample_run))
        lo, hi = builder.get_history_bounds()
        assert len(lo) == 32
        assert len(hi) == 32
        assert all(lo <= hi)


class TestPCAProjection:
    def test_pca_project_reduces_dimensions(self):
        vectors = np.random.RandomState(42).rand(20, 32)
        projected, pca_model = pca_project(vectors, n_components=6)
        assert projected.shape == (20, 6)

    def test_project_point_pca_consistent(self):
        vectors = np.random.RandomState(42).rand(20, 32)
        projected, pca_model = pca_project(vectors, n_components=6)
        # Project the first vector individually — should match batch result
        single = project_point_pca(vectors[0], pca_model)
        np.testing.assert_array_almost_equal(single, projected[0])
