"""Unit tests for src/oselmk/utils/kernels.py."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from oselmk.utils.kernels import SUPPORTED_KERNELS, kernel_matrix

# ---------------------------------------------------------------------------
# Fixtures / shared data
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N, D = 10, 3
A = RNG.standard_normal((N, D))
B = RNG.standard_normal((N, D))


# ---------------------------------------------------------------------------
# kernel_matrix — shape contract
# ---------------------------------------------------------------------------

def test_kernel_matrix_shape_square():
    K = kernel_matrix(A, A, kernel="rbf")
    assert K.shape == (N, N)

def test_kernel_matrix_shape_rectangular():
    m, n = 5, 8
    K = kernel_matrix(RNG.standard_normal((m, D)), RNG.standard_normal((n, D)))
    assert K.shape == (m, n)

def test_kernel_matrix_1d_inputs():
    a = RNG.standard_normal(N)
    b = RNG.standard_normal(N)
    K = kernel_matrix(a, b, kernel="rbf")
    assert K.shape == (N, N)


# ---------------------------------------------------------------------------
# RBF kernel
# ---------------------------------------------------------------------------

def test_rbf_diagonal_ones():
    """K(x, x) = exp(0) = 1 for all x."""
    K = kernel_matrix(A, A, kernel="rbf", params=1.0)
    assert np.allclose(np.diag(K), 1.0)

def test_rbf_symmetric():
    K = kernel_matrix(A, A, kernel="rbf")
    assert np.allclose(K, K.T)

def test_rbf_values_in_0_1():
    K = kernel_matrix(A, B, kernel="rbf")
    assert np.all(K >= 0) and np.all(K <= 1)

def test_rbf_sigma_effect():
    K1 = kernel_matrix(A, B, kernel="rbf", params=0.1)
    K2 = kernel_matrix(A, B, kernel="rbf", params=10.0)
    # Smaller sigma -> smaller values for x != y
    assert K1.mean() < K2.mean()

def test_rbf_invalid_sigma_raises():
    with pytest.raises(ValueError):
        kernel_matrix(A, B, kernel="rbf", params=0.0)


# ---------------------------------------------------------------------------
# Linear kernel
# ---------------------------------------------------------------------------

def test_linear_symmetric():
    K = kernel_matrix(A, A, kernel="linear")
    assert np.allclose(K, K.T)

def test_linear_known_value():
    a = np.array([[1.0, 0.0]])
    b = np.array([[2.0, 3.0]])
    K = kernel_matrix(a, b, kernel="linear")
    assert K[0, 0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Polynomial kernel
# ---------------------------------------------------------------------------

def test_poly_symmetric():
    K = kernel_matrix(A, A, kernel="poly", params=[1.0, 2.0])
    assert np.allclose(K, K.T)

def test_poly_known_value():
    a = np.array([[1.0, 1.0]])
    b = np.array([[1.0, 1.0]])
    # (1*1 + 1*1 + 1)^2 = (2+1)^2 = 9
    K = kernel_matrix(a, b, kernel="poly", params=[1.0, 2.0])
    assert K[0, 0] == pytest.approx(9.0)

def test_poly_missing_params_raises():
    with pytest.raises(ValueError):
        kernel_matrix(A, B, kernel="poly", params=[1.0])  # needs 2


# ---------------------------------------------------------------------------
# Wavelet kernel
# ---------------------------------------------------------------------------

def test_wavelet_symmetric():
    K = kernel_matrix(A, A, kernel="wavelet", params=[1.0, 1.0, 1.0])
    assert np.allclose(K, K.T)

def test_wavelet_invalid_a_raises():
    with pytest.raises(ValueError):
        kernel_matrix(A, B, kernel="wavelet", params=[1.0, 0.0, 1.0])

def test_wavelet_invalid_b_raises():
    with pytest.raises(ValueError):
        kernel_matrix(A, B, kernel="wavelet", params=[0.0, 1.0, 1.0])

def test_wavelet_missing_params_raises():
    with pytest.raises(ValueError):
        kernel_matrix(A, B, kernel="wavelet", params=[1.0, 1.0])  # needs 3

def test_wavelet_diagonal_same_point():
    """K(x, x): dist=0 -> cos(0)*exp(0) = 1 for all x."""
    K = kernel_matrix(A, A, kernel="wavelet", params=[1.0, 1.0, 1.0])
    assert np.allclose(np.diag(K), 1.0)


# ---------------------------------------------------------------------------
# Dispatcher: unknown kernel
# ---------------------------------------------------------------------------

def test_unknown_kernel_raises():
    with pytest.raises(ValueError, match="Unknown kernel"):
        kernel_matrix(A, B, kernel="sigmoid")

def test_supported_kernels_constant():
    assert set(SUPPORTED_KERNELS) == {"rbf", "linear", "poly", "wavelet"}
