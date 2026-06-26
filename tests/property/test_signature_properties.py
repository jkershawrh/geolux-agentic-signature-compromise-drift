import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from engine.geometric.distance import (
    cosine_similarity,
    euclidean_distance,
    geodesic_distance,
)
from engine.geometric.riemannian import compute_metric_tensor


float_vector = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=3, max_size=35,
)


class TestDistanceProperties:
    @given(float_vector)
    @settings(max_examples=100)
    def test_euclidean_self_distance_is_zero(self, v):
        arr = np.array(v)
        assert euclidean_distance(arr, arr) == pytest.approx(0.0, abs=1e-10)

    @given(float_vector, float_vector)
    @settings(max_examples=100)
    def test_euclidean_non_negative(self, v1, v2):
        assume(len(v1) == len(v2))
        a, b = np.array(v1), np.array(v2)
        assert euclidean_distance(a, b) >= 0

    @given(float_vector, float_vector)
    @settings(max_examples=100)
    def test_euclidean_symmetric(self, v1, v2):
        assume(len(v1) == len(v2))
        a, b = np.array(v1), np.array(v2)
        assert euclidean_distance(a, b) == pytest.approx(euclidean_distance(b, a))

    @given(st.integers(min_value=3, max_value=10).flatmap(
        lambda n: st.tuples(
            st.lists(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False), min_size=n, max_size=n),
            st.lists(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False), min_size=n, max_size=n),
            st.lists(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False), min_size=n, max_size=n),
        )
    ))
    @settings(max_examples=50)
    def test_triangle_inequality(self, vecs):
        v1, v2, v3 = vecs
        a, b, c = np.array(v1), np.array(v2), np.array(v3)
        d_ab = euclidean_distance(a, b)
        d_bc = euclidean_distance(b, c)
        d_ac = euclidean_distance(a, c)
        assert d_ac <= d_ab + d_bc + 1e-10

    @given(float_vector)
    @settings(max_examples=100)
    def test_cosine_self_similarity_is_one(self, v):
        arr = np.array(v)
        assume(np.linalg.norm(arr) > 1e-10)
        assert cosine_similarity(arr, arr) == pytest.approx(1.0, abs=1e-10)

    @given(float_vector, float_vector)
    @settings(max_examples=100)
    def test_cosine_in_range(self, v1, v2):
        assume(len(v1) == len(v2))
        a, b = np.array(v1), np.array(v2)
        sim = cosine_similarity(a, b)
        assert -1.0 - 1e-10 <= sim <= 1.0 + 1e-10

    @given(float_vector)
    @settings(max_examples=100)
    def test_geodesic_self_distance_is_zero(self, v):
        arr = np.array(v)
        assert geodesic_distance(arr, arr) == pytest.approx(0.0, abs=1e-10)

    @given(float_vector, float_vector)
    @settings(max_examples=100)
    def test_geodesic_non_negative(self, v1, v2):
        assume(len(v1) == len(v2))
        a, b = np.array(v1), np.array(v2)
        assert geodesic_distance(a, b) >= 0


class TestMetricTensorProperties:
    @given(st.integers(min_value=3, max_value=10))
    @settings(max_examples=20)
    def test_metric_tensor_is_symmetric(self, n):
        rng = np.random.RandomState(42)
        cov = rng.rand(n, n)
        cov = cov @ cov.T + np.eye(n)
        tensor = compute_metric_tensor(cov)
        np.testing.assert_array_almost_equal(tensor, tensor.T)

    @given(st.integers(min_value=3, max_value=10))
    @settings(max_examples=20)
    def test_metric_tensor_positive_definite(self, n):
        rng = np.random.RandomState(42)
        cov = rng.rand(n, n)
        cov = cov @ cov.T + np.eye(n)
        tensor = compute_metric_tensor(cov)
        eigenvalues = np.linalg.eigvalsh(tensor)
        assert all(eigenvalues > 0)
