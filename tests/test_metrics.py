"""Unit tests for src/oselmk/utils/metrics.py."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from oselmk.utils.metrics import rmse, nrmse, mape, smape, compute_all

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Perfect prediction -> all metrics = 0
Y_TRUE = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
Y_PRED_PERFECT = Y_TRUE.copy()

# Known error: pred = true + 1 always -> e=1, MSE=1, RMSE=1
Y_PRED_PLUS1 = Y_TRUE + 1.0

# Larger set for statistical tests
RNG = np.random.default_rng(7)
Y_LARGE = RNG.uniform(1.0, 10.0, 200)
Y_LARGE_PRED = Y_LARGE + RNG.normal(0, 0.5, 200)


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------


def test_rmse_perfect_prediction():
    """RMSE must be 0 for perfect predictions."""
    assert rmse(Y_TRUE, Y_PRED_PERFECT) == 0.0


def test_rmse_known_value():
    """RMSE for constant error of 1 must equal 1.0."""
    assert_allclose(rmse(Y_TRUE, Y_PRED_PLUS1), 1.0, atol=1e-12)


def test_rmse_non_negative():
    """RMSE must always be >= 0."""
    assert rmse(Y_LARGE, Y_LARGE_PRED) >= 0.0


def test_rmse_known_manual():
    """Manually verify: y_true=[3,3], y_pred=[4,5] -> RMSE=sqrt(2.5)."""
    y_t = np.array([3.0, 3.0])
    y_p = np.array([4.0, 5.0])  # errors: 1, 2 -> MSE=2.5
    assert_allclose(rmse(y_t, y_p), np.sqrt(2.5), atol=1e-12)


# ---------------------------------------------------------------------------
# NRMSE
# ---------------------------------------------------------------------------


def test_nrmse_known_value():
    """NRMSE = RMSE / range. range([1..5])=4, RMSE=1 -> NRMSE=0.25."""
    assert_allclose(nrmse(Y_TRUE, Y_PRED_PLUS1), 0.25, atol=1e-12)


def test_nrmse_perfect_prediction():
    """NRMSE must be 0 for perfect predictions."""
    assert nrmse(Y_TRUE, Y_PRED_PERFECT) == 0.0


def test_nrmse_constant_series_returns_nan():
    """NRMSE on a constant series must return nan and emit UserWarning."""
    y_const = np.ones(5)
    with pytest.warns(UserWarning, match="zero"):
        result = nrmse(y_const, y_const + 1.0)
    assert np.isnan(result)


def test_nrmse_less_than_rmse_for_wide_range():
    """For a series with range > 1, NRMSE < RMSE."""
    # range = 4, RMSE = 1  -> NRMSE = 0.25 < 1
    assert nrmse(Y_TRUE, Y_PRED_PLUS1) < rmse(Y_TRUE, Y_PRED_PLUS1)


# ---------------------------------------------------------------------------
# MAPE
# ---------------------------------------------------------------------------


def test_mape_perfect_prediction():
    """MAPE must be 0 for perfect predictions."""
    assert mape(Y_TRUE, Y_PRED_PERFECT) == 0.0


def test_mape_known_value():
    """MAPE for y_true=[1,2,4], y_pred=[2,2,4] -> mean([1, 0, 0]) = 1/3."""
    y_t = np.array([1.0, 2.0, 4.0])
    y_p = np.array([2.0, 2.0, 4.0])  # errors: 1/1=1, 0/2=0, 0/4=0
    assert_allclose(mape(y_t, y_p), 1.0 / 3.0, atol=1e-12)


def test_mape_excludes_zero_true_values():
    """Samples with y_true==0 must be excluded and trigger a UserWarning."""
    y_t = np.array([0.0, 2.0, 4.0])
    y_p = np.array([1.0, 2.0, 4.0])  # only 2 valid samples, both perfect
    with pytest.warns(UserWarning, match="near zero"):
        result = mape(y_t, y_p)
    assert_allclose(result, 0.0, atol=1e-12)


def test_mape_all_zero_returns_nan():
    """If all y_true are zero, MAPE must return nan."""
    with pytest.warns(UserWarning):
        result = mape(np.zeros(5), np.ones(5))
    assert np.isnan(result)


def test_mape_near_zero_excluded_via_isclose():
    """Near-zero (not exactly zero) y_true values must also be excluded."""
    y_t = np.array([1e-15, 2.0, 4.0])  # 1e-15 is near zero per np.isclose
    y_p = np.array([1.0, 2.0, 4.0])
    with pytest.warns(UserWarning, match="near zero"):
        result = mape(y_t, y_p)  # only 2 valid samples, both perfect
    assert_allclose(result, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# SMAPE
# ---------------------------------------------------------------------------


def test_smape_perfect_prediction():
    """SMAPE must be 0 for perfect predictions."""
    assert smape(Y_TRUE, Y_PRED_PERFECT) == 0.0


def test_smape_bounded():
    """SMAPE must be in [0, 2] for any inputs."""
    result = smape(Y_LARGE, Y_LARGE_PRED)
    assert 0.0 <= result <= 2.0


def test_smape_symmetry():
    """SMAPE(y_true, y_pred) must equal SMAPE(y_pred, y_true)."""
    assert_allclose(
        smape(Y_LARGE, Y_LARGE_PRED),
        smape(Y_LARGE_PRED, Y_LARGE),
        atol=1e-12,
    )


def test_smape_known_value():
    """Manual: y_true=[2], y_pred=[4] -> 2*|2-4|/(2+4) = 2*2/6 = 2/3."""
    y_t = np.array([2.0])
    y_p = np.array([4.0])
    assert_allclose(smape(y_t, y_p), 2.0 / 3.0, atol=1e-12)


def test_smape_excludes_double_zero():
    """Sample where both y_true and y_pred are 0 must be excluded."""
    y_t = np.array([0.0, 2.0])
    y_p = np.array([0.0, 4.0])  # first sample excluded; second: 2*2/6=2/3
    with pytest.warns(UserWarning, match="near zero"):
        result = smape(y_t, y_p)
    assert_allclose(result, 2.0 / 3.0, atol=1e-12)


def test_smape_all_double_zero_returns_nan():
    """If all denominators are zero, SMAPE must return nan."""
    with pytest.warns(UserWarning):
        result = smape(np.zeros(4), np.zeros(4))
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------


def test_compute_all_returns_dict_with_all_keys():
    """compute_all must return a dict with exactly the four metric keys."""
    result = compute_all(Y_TRUE, Y_PRED_PLUS1)
    assert set(result.keys()) == {"rmse", "nrmse", "mape", "smape"}


def test_compute_all_values_match_individual_functions():
    """Values in compute_all dict must match calling each function individually."""
    result = compute_all(Y_LARGE, Y_LARGE_PRED)
    assert_allclose(result["rmse"], rmse(Y_LARGE, Y_LARGE_PRED), atol=1e-12)
    assert_allclose(result["nrmse"], nrmse(Y_LARGE, Y_LARGE_PRED), atol=1e-12)
    assert_allclose(result["mape"], mape(Y_LARGE, Y_LARGE_PRED), atol=1e-12)
    assert_allclose(result["smape"], smape(Y_LARGE, Y_LARGE_PRED), atol=1e-12)


def test_compute_all_perfect_prediction_all_zeros():
    """All metrics must be 0 for perfect predictions."""
    result = compute_all(Y_TRUE, Y_PRED_PERFECT)
    assert result["rmse"] == 0.0
    assert result["nrmse"] == 0.0
    assert result["mape"] == 0.0
    assert result["smape"] == 0.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_mismatched_shapes_raise():
    """Mismatched array shapes must raise ValueError."""
    with pytest.raises(ValueError, match="same shape"):
        rmse(np.ones(5), np.ones(3))


def test_empty_arrays_raise():
    """Empty arrays must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        rmse(np.array([]), np.array([]))
