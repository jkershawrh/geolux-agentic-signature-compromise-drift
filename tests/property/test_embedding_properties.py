import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from engine.geometric.embedding import (
    aggregate_metric_vectors,
    normalize_vector,
)

float_vector = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=3, max_size=36,
)


class TestAggregationProperties:
    @given(float_vector)
    @settings(max_examples=100)
    def test_aggregate_single_is_identity(self, v):
        arr = np.array(v)
        result = aggregate_metric_vectors([arr])
        np.testing.assert_array_almost_equal(result, arr)

    @given(float_vector)
    @settings(max_examples=100)
    def test_aggregate_identical_is_identity(self, v):
        arr = np.array(v)
        result = aggregate_metric_vectors([arr, arr, arr])
        np.testing.assert_array_almost_equal(result, arr)

    @given(float_vector, float_vector)
    @settings(max_examples=100)
    def test_aggregate_result_between_inputs(self, v1, v2):
        assume(len(v1) == len(v2))
        a, b = np.array(v1), np.array(v2)
        result = aggregate_metric_vectors([a, b])
        for i in range(len(result)):
            assert min(a[i], b[i]) - 1e-10 <= result[i] <= max(a[i], b[i]) + 1e-10


class TestNormalizationProperties:
    @given(float_vector)
    @settings(max_examples=100)
    def test_normalize_produces_unit_range(self, v):
        arr = np.array(v)
        lo = np.zeros_like(arr)
        hi = np.ones_like(arr)
        result = normalize_vector(arr, lo, hi)
        for val in result:
            assert -1e-10 <= val <= 1.0 + 1e-10
