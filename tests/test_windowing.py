"""Unit tests for src/oselmk/utils/windowing.py."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from oselmk.utils.windowing import make_lag_features

# ---------------------------------------------------------------------------
# Basic shape and content
# ---------------------------------------------------------------------------


def test_output_shape_default_lags():
    """With n_lags=1, X has 1 feature column and y has n-1 rows."""
    ts = np.arange(10, dtype=float)
    X, y = make_lag_features(ts, n_lags=1)
    assert X.shape == (9, 1)
    assert y.shape == (9,)


def test_output_shape_multiple_lags():
    ts = np.arange(20, dtype=float)
    X, y = make_lag_features(ts, n_lags=4)
    assert X.shape == (16, 4)
    assert y.shape == (16,)


def test_values_single_lag():
    """y[i] == ts[i+1] and X[i, 0] == ts[i]."""
    ts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X, y = make_lag_features(ts, n_lags=1)
    assert_array_equal(y, [2.0, 3.0, 4.0, 5.0])
    assert_array_equal(X[:, 0], [1.0, 2.0, 3.0, 4.0])


def test_values_two_lags():
    """With n_lags=2: X[i] = [ts[i+1], ts[i]], y[i] = ts[i+2]."""
    ts = np.arange(6, dtype=float)
    X, y = make_lag_features(ts, n_lags=2)
    assert_array_equal(y, [2.0, 3.0, 4.0, 5.0])
    assert_array_equal(X[:, 0], [1.0, 2.0, 3.0, 4.0])  # lag-1
    assert_array_equal(X[:, 1], [0.0, 1.0, 2.0, 3.0])  # lag-2


def test_no_copy_of_original_series():
    """The function must not mutate the input array."""
    ts = np.arange(10, dtype=float)
    original = ts.copy()
    make_lag_features(ts, n_lags=2)
    assert_array_equal(ts, original)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_minimum_valid_length():
    """A series of length n_lags+1 yields exactly one sample."""
    ts = np.array([1.0, 2.0, 3.0])
    X, y = make_lag_features(ts, n_lags=2)
    assert X.shape == (1, 2)
    assert y.shape == (1,)


def test_raises_on_series_too_short():
    """Series shorter than n_lags+1 must raise ValueError."""
    with pytest.raises(ValueError, match="n_lags"):
        make_lag_features(np.array([1.0, 2.0]), n_lags=2)


def test_raises_on_non_positive_lags():
    with pytest.raises(ValueError, match="n_lags"):
        make_lag_features(np.arange(10, dtype=float), n_lags=0)


def test_raises_on_2d_input():
    with pytest.raises(ValueError, match="1-D"):
        make_lag_features(np.ones((5, 2)), n_lags=1)


def test_raises_on_empty_series():
    with pytest.raises(ValueError):
        make_lag_features(np.array([]), n_lags=1)


# ---------------------------------------------------------------------------
# dtype preservation
# ---------------------------------------------------------------------------


def test_float32_input_preserved():
    ts = np.arange(10, dtype=np.float32)
    X, y = make_lag_features(ts, n_lags=2)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_integer_input_preserved():
    ts = np.arange(10, dtype=int)
    X, y = make_lag_features(ts, n_lags=2)
    assert X.dtype == int
    assert y.dtype == int


# ---------------------------------------------------------------------------
# Large series smoke test
# ---------------------------------------------------------------------------


def test_large_series_smoke():
    ts = np.random.default_rng(99).standard_normal(10_000)
    X, y = make_lag_features(ts, n_lags=10)
    assert X.shape == (9990, 10)
    assert y.shape == (9990,)
