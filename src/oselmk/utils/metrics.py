"""Regression metrics for time series prediction evaluation.

All functions follow the scikit-learn convention: they receive the raw
arrays ``(y_true, y_pred)`` instead of a pre-computed error vector.

This fixes a semantic ambiguity in the original Octave code where ``mse.m``
expected the **error vector** ``e = target - output`` rather than the raw
predictions.  Here every function is self-contained.

Metrics
-------

``rmse``
    Root Mean Squared Error.  Always non-negative; 0 for perfect predictions.

``nrmse``
    Normalised RMSE: ``rmse / (max(y_true) - min(y_true))``.
    Returns ``nan`` when the series is constant (zero range).

``mape``
    Mean Absolute Percentage Error.
    Near-zero true values (|y| < 1e-8) are excluded to avoid division by
    zero, matching the guard in the original ``calc_errors.m``.
    Returns ``nan`` when *all* true values are near zero.

``smape``
    Symmetric MAPE, bounded in ``[0, 2]``.
    Uses ``(|y_true| + |y_pred|) / 2`` as denominator; pairs where the
    denominator is zero contribute 0 to the mean.

``compute_all``
    Aggregator: returns a ``dict`` with all four metrics.
"""

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate(y_true: NDArray, y_pred: NDArray) -> tuple[NDArray, NDArray]:
    """Coerce inputs and enforce shape compatibility."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true has {y_true.shape}, y_pred has {y_pred.shape}."
        )
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------


def rmse(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n,)

    Returns
    -------
    float
        Always >= 0.  Equal to 0 only for perfect predictions.
    """
    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> float:
    """Normalised RMSE (range-normalised).

    Returns
    -------
    float
        ``rmse / (max(y_true) - min(y_true))``.
        Returns ``nan`` for constant series.
    """
    y_true, y_pred = _validate(y_true, y_pred)
    r = float(y_true.max() - y_true.min())
    if r == 0.0:
        warnings.warn(
            "nrmse: y_true is constant (range=0); returning nan.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    return rmse(y_true, y_pred) / r


def mape(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    eps: float = 1e-8,
) -> float:
    """Mean Absolute Percentage Error.

    Near-zero true values (|y_true| < eps) are excluded.

    Returns
    -------
    float
        Value in [0, inf).  Returns ``nan`` if all true values are near zero.
    """
    y_true, y_pred = _validate(y_true, y_pred)
    mask = np.abs(y_true) >= eps
    if not np.any(mask):
        warnings.warn(
            "mape: all y_true values are near zero; returning nan.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def smape(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> float:
    """Symmetric Mean Absolute Percentage Error, bounded in [0, 2].

    Denominator is ``(|y_true| + |y_pred|) / 2``.
    Pairs where the denominator is zero contribute 0.

    Returns
    -------
    float
        Value in [0, 2].
    """
    y_true, y_pred = _validate(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    numer = np.abs(y_true - y_pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(denom == 0.0, 0.0, numer / denom)
    return float(np.mean(ratio))


def compute_all(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> dict[str, Any]:
    """Compute all four metrics in one call.

    Returns
    -------
    dict
        Keys: ``'rmse'``, ``'nrmse'``, ``'mape'``, ``'smape'``.
    """
    return {
        "rmse":  rmse(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "mape":  mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
