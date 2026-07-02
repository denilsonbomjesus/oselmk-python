"""Unit tests for src/oselmk/utils/metrics.py."""

import numpy as np
import pytest

from oselmk.utils.metrics import compute_all, mape, nrmse, rmse, smape

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
N = 50
y_true = RNG.uniform(1.0, 10.0, N)
y_pred = y_true + RNG.normal(0, 0.1, N)


# ---------------------------------------------------------------------------
# rmse
# ---------------------------------------------------------------------------

def test_rmse_perfect_prediction():
    assert rmse(y_true, y_true) == pytest.approx(0.0, abs=1e-12)

def test_rmse_known_value():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([2.0, 2.0, 2.0])
    assert rmse(yt, yp) == pytest.approx((2.0 / 3.0) ** 0.5, rel=1e-9)

def test_rmse_non_negative():
    assert rmse(y_true, y_pred) >= 0.0

def test_rmse_shape_mismatch_raises():
    with pytest.raises(ValueError):
        rmse(y_true, y_pred[:10])

def test_rmse_empty_raises():
    with pytest.raises(ValueError):
        rmse(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# nrmse
# ---------------------------------------------------------------------------

def test_nrmse_perfect_prediction():
    assert nrmse(y_true, y_true) == pytest.approx(0.0, abs=1e-12)

def test_nrmse_non_negative():
    assert nrmse(y_true, y_pred) >= 0.0

def test_nrmse_constant_series_returns_nan():
    yt = np.ones(10)
    yp = np.ones(10) * 1.5
    assert np.isnan(nrmse(yt, yp))

def test_nrmse_normalised_by_range():
    yt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    yp = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
    r = rmse(yt, yp)
    expected = r / (4.0 - 0.0)
    assert nrmse(yt, yp) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# mape
# ---------------------------------------------------------------------------

def test_mape_perfect_prediction():
    assert mape(y_true, y_true) == pytest.approx(0.0, abs=1e-12)

def test_mape_non_negative():
    assert mape(y_true, y_pred) >= 0.0

def test_mape_known_value():
    yt = np.array([100.0, 200.0])
    yp = np.array([110.0, 190.0])
    assert mape(yt, yp) == pytest.approx(0.075, rel=1e-9)

def test_mape_all_zeros_returns_nan():
    yt = np.zeros(5)
    yp = np.ones(5)
    assert np.isnan(mape(yt, yp))


# ---------------------------------------------------------------------------
# smape
# ---------------------------------------------------------------------------

def test_smape_perfect_prediction():
    assert smape(y_true, y_true) == pytest.approx(0.0, abs=1e-12)

def test_smape_non_negative():
    assert smape(y_true, y_pred) >= 0.0

def test_smape_bounded_by_2():
    yt = np.array([1.0, 0.0, -1.0])
    yp = np.array([100.0, 100.0, 100.0])
    assert smape(yt, yp) <= 2.0 + 1e-9

def test_smape_known_value():
    yt = np.array([100.0])
    yp = np.array([200.0])
    assert smape(yt, yp) == pytest.approx(100.0 / 150.0, rel=1e-9)


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------

def test_compute_all_returns_dict():
    result = compute_all(y_true, y_pred)
    assert isinstance(result, dict)

def test_compute_all_keys():
    result = compute_all(y_true, y_pred)
    assert set(result.keys()) == {"rmse", "nrmse", "mape", "smape"}

def test_compute_all_values_match_individual():
    result = compute_all(y_true, y_pred)
    assert result["rmse"]  == pytest.approx(rmse(y_true, y_pred),  rel=1e-9)
    assert result["nrmse"] == pytest.approx(nrmse(y_true, y_pred), rel=1e-9)
    assert result["mape"]  == pytest.approx(mape(y_true, y_pred),  rel=1e-9)
    assert result["smape"] == pytest.approx(smape(y_true, y_pred), rel=1e-9)
