"""Unit tests for src/oselmk/kernels.py."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from oselmk.kernels import kernel_matrix, SUPPORTED_KERNELS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
X_TRAIN = RNG.standard_normal((10, 3))
X_TEST = RNG.standard_normal((5, 3))


# ---------------------------------------------------------------------------
# Generic properties (parametrised over all kernels)
# ---------------------------------------------------------------------------

KERNEL_PARAMS = {
    "rbf": 1.0,
    "linear": 1.0,
    "polynomial": [0.0, 2],
    "wavelet": [1.0, 1.0, 1.0],
}


@pytest.mark.parametrize("ktype", SUPPORTED_KERNELS)
def test_omega_train_shape(ktype):
    """Omega_train must be square: (n_train, n_train)."""
    omega = kernel_matrix(X_TRAIN, ktype, KERNEL_PARAMS[ktype])
    assert omega.shape == (len(X_TRAIN), len(X_TRAIN))


@pytest.mark.parametrize("ktype", SUPPORTED_KERNELS)
def test_omega_test_shape(ktype):
    """Omega_test must be (n_train, n_test)."""
    omega = kernel_matrix(X_TRAIN, ktype, KERNEL_PARAMS[ktype], X_TEST)
    assert omega.shape == (len(X_TRAIN), len(X_TEST))


@pytest.mark.parametrize("ktype", SUPPORTED_KERNELS)
def test_omega_train_symmetry(ktype):
    """Omega_train must be symmetric: K(x_i, x_j) == K(x_j, x_i)."""
    omega = kernel_matrix(X_TRAIN, ktype, KERNEL_PARAMS[ktype])
    assert_allclose(omega, omega.T, atol=1e-12)


# ---------------------------------------------------------------------------
# RBF-specific tests
# ---------------------------------------------------------------------------


def test_rbf_diagonal_is_one():
    """RBF(x, x) = exp(0) = 1 for all x."""
    omega = kernel_matrix(X_TRAIN, "rbf", 1.0)
    assert_allclose(np.diag(omega), np.ones(len(X_TRAIN)), atol=1e-12)


def test_rbf_all_positive():
    """RBF kernel values are strictly positive."""
    omega = kernel_matrix(X_TRAIN, "rbf", 1.0)
    assert np.all(omega > 0)


def test_rbf_positive_semidefinite():
    """RBF Gram matrix must be positive semi-definite (eigenvalues >= 0)."""
    omega = kernel_matrix(X_TRAIN, "rbf", 1.0)
    eigenvalues = np.linalg.eigvalsh(omega)
    assert np.all(eigenvalues >= -1e-10)


def test_rbf_invalid_sigma():
    """Negative or zero sigma must raise ValueError."""
    with pytest.raises(ValueError, match="sigma > 0"):
        kernel_matrix(X_TRAIN, "rbf", -1.0)


def test_rbf_larger_sigma_smoother():
    """Larger sigma -> values closer to 1 (wider Gaussian)."""
    omega_tight = kernel_matrix(X_TRAIN, "rbf", 0.01)
    omega_wide = kernel_matrix(X_TRAIN, "rbf", 1000.0)
    # Off-diagonal values should be higher for wider kernel
    off_tight = omega_tight[0, 1:].mean()
    off_wide = omega_wide[0, 1:].mean()
    assert off_wide > off_tight


# ---------------------------------------------------------------------------
# Linear kernel tests
# ---------------------------------------------------------------------------


def test_linear_equivalence_with_dot_product():
    """Linear kernel matrix == X_train @ X_train.T."""
    omega = kernel_matrix(X_TRAIN, "linear", 1.0)
    expected = X_TRAIN @ X_TRAIN.T
    assert_allclose(omega, expected, atol=1e-12)


def test_linear_cross_equivalence():
    """Linear cross-kernel == X_train @ X_test.T."""
    omega = kernel_matrix(X_TRAIN, "linear", 1.0, X_TEST)
    expected = X_TRAIN @ X_TEST.T
    assert_allclose(omega, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Polynomial kernel tests
# ---------------------------------------------------------------------------


def test_polynomial_degree2_c0_equals_linear_squared():
    """poly(x, y; c=0, d=2) = (x.T y)^2."""
    omega_poly = kernel_matrix(X_TRAIN, "polynomial", [0.0, 2])
    omega_lin = kernel_matrix(X_TRAIN, "linear", 1.0)
    assert_allclose(omega_poly, omega_lin ** 2, atol=1e-10)


def test_polynomial_missing_params_raises():
    """Polynomial kernel with only one param must raise ValueError."""
    with pytest.raises(ValueError, match="params=\\[c, d\\]"):
        kernel_matrix(X_TRAIN, "polynomial", 2.0)


# ---------------------------------------------------------------------------
# Wavelet kernel tests
# ---------------------------------------------------------------------------


def test_wavelet_missing_params_raises():
    """Wavelet kernel with fewer than 3 params must raise ValueError."""
    with pytest.raises(ValueError, match="params=\\[b, a, w0\\]"):
        kernel_matrix(X_TRAIN, "wavelet", [1.0, 1.0])


def test_wavelet_symmetric():
    """Wavelet Omega_train must be symmetric."""
    omega = kernel_matrix(X_TRAIN, "wavelet", [1.0, 1.0, 1.0])
    assert_allclose(omega, omega.T, atol=1e-12)


# ---------------------------------------------------------------------------
# Unknown kernel
# ---------------------------------------------------------------------------


def test_unknown_kernel_raises():
    """Unsupported kernel name must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown kernel"):
        kernel_matrix(X_TRAIN, "sigmoid", 1.0)


# ---------------------------------------------------------------------------
# 1-D input auto-reshape
# ---------------------------------------------------------------------------


def test_1d_input_is_accepted():
    """1-D arrays should be silently reshaped to (n, 1)."""
    x = RNG.standard_normal(8)
    omega = kernel_matrix(x, "rbf", 1.0)
    assert omega.shape == (8, 8)
