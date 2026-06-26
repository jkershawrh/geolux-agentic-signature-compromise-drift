import numpy as np
import pytest

from engine.geometric.distance import (
    cosine_similarity,
    drift_direction,
    euclidean_distance,
    frechet_mean,
    geodesic_distance,
    per_dimension_distances,
)


class TestEuclideanDistance:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance(v, v) == 0.0

    def test_known_distance(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        assert euclidean_distance(v1, v2) == pytest.approx(5.0)

    def test_symmetric(self):
        v1 = np.array([1.0, 2.0])
        v2 = np.array([4.0, 6.0])
        assert euclidean_distance(v1, v2) == pytest.approx(euclidean_distance(v2, v1))

    def test_non_negative(self):
        v1 = np.array([1.0, -2.0, 3.0])
        v2 = np.array([-1.0, 2.0, -3.0])
        assert euclidean_distance(v1, v2) >= 0


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 1.0])
        assert cosine_similarity(v1, v2) == 0.0

    def test_range(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        sim = cosine_similarity(v1, v2)
        assert -1.0 <= sim <= 1.0


class TestGeodesicDistance:
    def test_without_metric_is_euclidean(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        assert geodesic_distance(v1, v2) == pytest.approx(5.0)

    def test_with_identity_metric_is_euclidean(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        metric = np.eye(2)
        assert geodesic_distance(v1, v2, metric) == pytest.approx(5.0)

    def test_with_scaled_metric(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 0.0])
        metric = np.array([[4.0, 0.0], [0.0, 1.0]])
        assert geodesic_distance(v1, v2, metric) == pytest.approx(2.0)

    def test_non_negative(self):
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        metric = np.eye(3) * 2
        assert geodesic_distance(v1, v2, metric) >= 0

    def test_identical_is_zero(self):
        v = np.array([1.0, 2.0])
        metric = np.array([[2.0, 0.5], [0.5, 3.0]])
        assert geodesic_distance(v, v, metric) == pytest.approx(0.0, abs=1e-10)


class TestFrechetMean:
    def test_single_vector(self):
        v = np.array([1.0, 2.0, 3.0])
        result = frechet_mean([v])
        np.testing.assert_array_almost_equal(result, v)

    def test_euclidean_mean(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([2.0, 2.0])
        result = frechet_mean([v1, v2])
        np.testing.assert_array_almost_equal(result, [1.0, 1.0])

    def test_with_metric_tensor(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([2.0, 2.0])
        metric = np.eye(2)
        result = frechet_mean([v1, v2], metric_tensor=metric)
        np.testing.assert_array_almost_equal(result, [1.0, 1.0], decimal=3)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            frechet_mean([])


class TestPerDimensionDistances:
    def test_correct_decomposition(self):
        a = np.zeros(32)
        b = np.zeros(32)
        b[0] = 1.0  # first metric in response_structure

        dim_sizes = [6, 4, 5, 4, 4, 3, 3, 3]
        result = per_dimension_distances(a, b, dim_sizes)

        assert "response_structure" in result
        assert result["response_structure"] == pytest.approx(1.0)
        assert result["token_economics"] == pytest.approx(0.0)

    def test_wrong_dimension_count_raises(self):
        with pytest.raises(ValueError):
            per_dimension_distances(np.zeros(32), np.zeros(32), [6, 4, 5])


class TestDriftDirection:
    def test_unit_vector(self):
        baseline = np.array([0.0, 0.0])
        current = np.array([3.0, 4.0])
        direction = drift_direction(baseline, current)
        assert np.linalg.norm(direction) == pytest.approx(1.0)
        np.testing.assert_array_almost_equal(direction, [0.6, 0.8])

    def test_no_drift_returns_zeros(self):
        v = np.array([1.0, 2.0])
        direction = drift_direction(v, v)
        np.testing.assert_array_almost_equal(direction, [0.0, 0.0])
