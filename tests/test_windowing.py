"""Unit tests for src/oselmk/utils/windowing.py."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from oselmk.utils.windowing import make_lag_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SERIES = np.arange(1.0, 11.0)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_shape_n_lags_1():
    """With n_lags=1: X shape (N-1, 1), y shape (N-1,)."""
    X, y = make_lag_features(SERIES, n_lags=1)
    assert X.shape == (len(SERIES) - 1, 1)
    assert y.shape == (len(SERIES) - 1,)


def test_shape_n_lags_3():
    """With n_lags=3: X shape (N-3, 3), y shape (N-3,)."""
    X, y = make_lag_features(SERIES, n_lags=3)
    assert X.shape == (len(SERIES) - 3, 3)
    assert y.shape == (len(SERIES) - 3,)


def test_shape_n_lags_equals_n_minus_1():
    """Maximum n_lags = N-1 must produce exactly 1 sample."""
    n = len(SERIES)
    X, y = make_lag_features(SERIES, n_lags=n - 1)
    assert X.shape == (1, n - 1)
    assert y.shape == (1,)


# ---------------------------------------------------------------------------
# Correctness of values — n_lags=1
# ---------------------------------------------------------------------------


def test_values_n_lags_1_y():
    """y must be series[1:] when n_lags=1."""
    _, y = make_lag_features(SERIES, n_lags=1)
    assert_array_equal(y, SERIES[1:])


def test_values_n_lags_1_X():
    """X column-0 must be series[:-1] when n_lags=1."""
    X, _ = make_lag_features(SERIES, n_lags=1)
    assert_array_equal(X[:, 0], SERIES[:-1])


def test_values_n_lags_1_pairs():
    """Each (X[i,0], y[i]) must be a consecutive pair (s_i, s_{i+1})."""
    X, y = make_lag_features(SERIES, n_lags=1)
    for i in range(len(y)):
        assert X[i, 0] == SERIES[i]
        assert y[i] == SERIES[i + 1]


# ---------------------------------------------------------------------------
# Correctness of values — n_lags=3
# ---------------------------------------------------------------------------


def test_values_n_lags_3_known_series():
    """Verify exact expected matrix for small known series with n_lags=2.

    series = [1, 2, 3, 4, 5]
    Expected X (lag-1 col-0, lag-2 col-1):
        row 0: [2, 1]  -> y=3
        row 1: [3, 2]  -> y=4
        row 2: [4, 3]  -> y=5
    """
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X, y = make_lag_features(s, n_lags=2)

    expected_X = np.array([[2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])
    expected_y = np.array([3.0, 4.0, 5.0])

    assert_array_equal(X, expected_X)
    assert_array_equal(y, expected_y)


def test_values_n_lags_3_y():
    """y must be series[3:] when n_lags=3."""
    _, y = make_lag_features(SERIES, n_lags=3)
    assert_array_equal(y, SERIES[3:])


def test_values_n_lags_3_col0_is_lag1():
    """Column 0 of X (lag-1) must equal series[2:-1] when n_lags=3."""
    X, _ = make_lag_features(SERIES, n_lags=3)
    # lag-1: one step before target -> series[2], series[3], ...
    assert_array_equal(X[:, 0], SERIES[2:-1])


def test_values_n_lags_3_col2_is_lag3():
    """Column 2 of X (lag-3) must equal series[:-3] when n_lags=3 (oldest)."""
    X, _ = make_lag_features(SERIES, n_lags=3)
    assert_array_equal(X[:, 2], SERIES[: len(SERIES) - 3])


def test_consecutive_rows_overlap_correctly():
    """Consecutive rows must share n_lags-1 values (sliding window property)."""
    X, _ = make_lag_features(SERIES, n_lags=3)
    for i in range(len(X) - 1):
        # row i+1's last (n_lags-1) lags == row i's first (n_lags-1) lags
        assert_array_equal(X[i + 1, 1:], X[i, :-1])


# ---------------------------------------------------------------------------
# Input type handling
# ---------------------------------------------------------------------------


def test_accepts_python_list():
    """Plain Python list must be accepted and produce correct output."""
    X, y = make_lag_features([1.0, 2.0, 3.0, 4.0], n_lags=1)
    assert X.shape == (3, 1)
    assert y.shape == (3,)


def test_output_dtype_float():
    """Output arrays must be float64 regardless of integer input."""
    X, y = make_lag_features([1, 2, 3, 4], n_lags=1)
    assert X.dtype == np.float64
    assert y.dtype == np.float64


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------


def test_2d_input_raises():
    """2-D input must raise ValueError."""
    with pytest.raises(ValueError, match="1-D"):
        make_lag_features(np.ones((5, 2)), n_lags=1)


def test_n_lags_zero_raises():
    """n_lags=0 must raise ValueError."""
    with pytest.raises(ValueError, match="n_lags.*>=.*1"):
        make_lag_features(SERIES, n_lags=0)


def test_n_lags_negative_raises():
    """Negative n_lags must raise ValueError."""
    with pytest.raises(ValueError, match="n_lags.*>=.*1"):
        make_lag_features(SERIES, n_lags=-2)


def test_series_too_short_raises():
    """Series shorter than n_lags + 1 must raise ValueError."""
    with pytest.raises(ValueError, match="at least"):
        make_lag_features([1.0, 2.0], n_lags=3)


def test_series_exactly_n_lags_plus_1():
    """A series of exactly n_lags+1 elements must produce exactly 1 sample."""
    s = np.array([10.0, 20.0, 30.0, 40.0])  # len=4, n_lags=3
    X, y = make_lag_features(s, n_lags=3)
    assert X.shape == (1, 3)
    assert y.shape == (1,)
    assert_array_equal(X[0], [30.0, 20.0, 10.0])
    assert y[0] == 40.0
