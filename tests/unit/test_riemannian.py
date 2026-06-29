import numpy as np
import pytest

from engine.geometric.riemannian import (
    compute_metric_tensor,
    compute_metric_tensor_shrinkage,
    local_metric_tensor,
    anisotropy_estimate,
)


class TestComputeMetricTensor:
    def test_identity_covariance_gives_identity(self):
        cov = np.eye(3)
        tensor = compute_metric_tensor(cov, regularization=0.0)
        np.testing.assert_array_almost_equal(tensor, np.eye(3))

    def test_diagonal_covariance(self):
        cov = np.diag([4.0, 1.0, 0.25])
        tensor = compute_metric_tensor(cov, regularization=0.0)
        np.testing.assert_array_almost_equal(tensor, np.diag([0.25, 1.0, 4.0]))

    def test_regularization_prevents_singular(self):
        cov = np.zeros((3, 3))
        tensor = compute_metric_tensor(cov, regularization=1e-6)
        assert not np.any(np.isnan(tensor))
        assert not np.any(np.isinf(tensor))

    def test_symmetric_output(self):
        cov = np.array([[2.0, 0.5], [0.5, 3.0]])
        tensor = compute_metric_tensor(cov)
        np.testing.assert_array_almost_equal(tensor, tensor.T)

    def test_positive_definite_output(self):
        cov = np.array([[2.0, 0.5], [0.5, 3.0]])
        tensor = compute_metric_tensor(cov)
        eigenvalues = np.linalg.eigvalsh(tensor)
        assert all(eigenvalues > 0)


class TestLocalMetricTensor:
    def test_local_metric_from_cluster(self):
        rng = np.random.RandomState(42)
        vectors = rng.randn(20, 5) * 0.1 + np.array([1, 2, 3, 4, 5])
        point = np.array([1, 2, 3, 4, 5])

        tensor = local_metric_tensor(vectors, point, k_neighbors=10)
        assert tensor.shape == (5, 5)
        np.testing.assert_array_almost_equal(tensor, tensor.T)

    def test_fewer_neighbors_than_k(self):
        vectors = np.array([[1, 2], [3, 4], [5, 6]])
        point = np.array([2, 3])
        tensor = local_metric_tensor(vectors, point, k_neighbors=100)
        assert tensor.shape == (2, 2)


class TestAnisotropyEstimate:
    def test_identity_metric_zero_anisotropy(self):
        metric = np.eye(5)
        anisotropy = anisotropy_estimate(metric)
        assert anisotropy == pytest.approx(0.0)

    def test_anisotropic_metric_positive_anisotropy(self):
        metric = np.diag([1, 10, 100, 1000, 10000])
        anisotropy = anisotropy_estimate(metric)
        assert anisotropy > 0


class TestLedoitWolfShrinkage:
    def test_ledoit_wolf_shrinkage(self):
        vectors = np.random.RandomState(42).randn(10, 35)
        tensor = compute_metric_tensor_shrinkage(vectors)
        assert tensor.shape == (35, 35)
        np.testing.assert_array_almost_equal(tensor, tensor.T)  # symmetric

    def test_single_sample_returns_identity(self):
        vectors = np.random.RandomState(0).randn(1, 5)
        tensor = compute_metric_tensor_shrinkage(vectors)
        np.testing.assert_array_almost_equal(tensor, np.eye(5))
