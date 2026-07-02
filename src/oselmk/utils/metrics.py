"""Regression metrics for time series prediction evaluation.

All functions follow the scikit-learn convention: they receive the raw
arrays ``(y_true, y_pred)`` instead of a pre-computed error vector.

This fixes a semantic ambiguity in the original Octave code where ``mse.m``
expected the *error* ``E = target - output`` instead of the raw arrays,
causing ``calc_errors.m`` to call ``sqrt(mse(target - output))`` -- correct
but confusing.  Here, every function is self-contained.

MAPE edge case
--------------
The original code used ``all(target != 0)`` to guard against division by
zero, which silently returns NaN for the whole batch if *any* true value is
exactly zero -- even floating-point rounding artefacts can trigger this.
Here, ``np.isclose(y_true, 0)`` is used to identify near-zero entries;
those samples are excluded from the MAPE average and a warning is issued
when exclusions occur.

Reference formulas
------------------
* RMSE  : sqrt(mean((y_true - y_pred)^2))
* NRMSE : RMSE / (max(y_true) - min(y_true))
* MAPE  : mean(|e_i / y_true_i|)  for y_true_i != 0
* SMAPE : 2 * mean(|e_i| / (|y_true_i| + |y_pred_i|))
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Validate and coerce inputs to 1-D float arrays of the same length."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape. "
            f"Got {y_true.shape} and {y_pred.shape}."
        )
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def rmse(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        Ground-truth values.
    y_pred : array-like, shape (n,)
        Predicted values.

    Returns
    -------
    float
        RMSE >= 0.  Returns 0.0 for perfect predictions.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> float:
    """Normalised Root Mean Squared Error (range-normalised).

    NRMSE = RMSE / (max(y_true) - min(y_true))

    Allows comparison of RMSE across series with different scales.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

    Returns
    -------
    float
        NRMSE >= 0.  Returns ``nan`` if the range of y_true is zero
        (constant series) and emits a :class:`UserWarning`.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    r = float(np.max(y_true) - np.min(y_true))
    if np.isclose(r, 0.0):
        warnings.warn(
            "Range of y_true is zero (constant series). NRMSE is undefined (nan).",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")
    return rmse(y_true, y_pred) / r


def mape(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> float:
    """Mean Absolute Percentage Error.

    MAPE = mean(|y_true_i - y_pred_i| / |y_true_i|)  for y_true_i != 0

    Samples where ``y_true`` is near zero (``np.isclose``) are excluded from
    the average.  A :class:`UserWarning` is issued when any samples are
    excluded.  Returns ``nan`` if *all* samples are excluded.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

    Returns
    -------
    float
        MAPE >= 0 (in absolute ratio units, not percentage).  Multiply by 100
        for the percentage form.  Returns ``nan`` if no valid samples remain.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    near_zero = np.isclose(y_true, 0.0)
    n_excluded = int(near_zero.sum())
    if n_excluded > 0:
        warnings.warn(
            f"{n_excluded} sample(s) with y_true near zero excluded from MAPE.",
            UserWarning,
            stacklevel=2,
        )
    valid = ~near_zero
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[valid] - y_pred[valid]) / np.abs(y_true[valid])))


def smape(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> float:
    """Symmetric Mean Absolute Percentage Error.

    SMAPE = 2 * mean(|y_true_i - y_pred_i| / (|y_true_i| + |y_pred_i|))

    Unlike MAPE, SMAPE is symmetric and bounded in [0, 2].  Samples where
    both ``y_true`` and ``y_pred`` are zero are excluded (denominator = 0)
    and a :class:`UserWarning` is issued.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

    Returns
    -------
    float
        SMAPE in [0, 2].  Returns ``nan`` if no valid samples remain.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    zero_denom = np.isclose(denominator, 0.0)
    n_excluded = int(zero_denom.sum())
    if n_excluded > 0:
        warnings.warn(
            f"{n_excluded} sample(s) where both y_true and y_pred are near "
            "zero excluded from SMAPE.",
            UserWarning,
            stacklevel=2,
        )
    valid = ~zero_denom
    if not valid.any():
        return float("nan")
    return float(
        2.0 * np.mean(np.abs(y_true[valid] - y_pred[valid]) / denominator[valid])
    )


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------


def compute_all(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> dict[str, float]:
    """Compute all four metrics in a single call.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

    Returns
    -------
    dict with keys ``'rmse'``, ``'nrmse'``, ``'mape'``, ``'smape'``.
    Each value is a float (may be ``nan`` in degenerate cases).
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
