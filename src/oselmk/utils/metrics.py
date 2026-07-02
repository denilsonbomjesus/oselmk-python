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
    Returns ``nan`` for constant series (zero range) to avoid division by
    zero—matching the behaviour of the Octave ``calc_errors.m`` guard.

``mape``
    Mean Absolute Percentage Error.  Excludes samples where
    ``|y_true| < eps`` (following the Octave ``if all(target != 0)`` guard)
    and returns ``nan`` when *all* true values are near zero.

``smape``
    Symmetric MAPE: ``mean(|y_true - y_pred| / ((|y_true|+|y_pred|)/2))``.
    Bounded in ``[0, 2]``; degenerate ``0/0`` samples contribute 0.

``compute_all``
    Convenience wrapper returning all four metrics as a ``dict``.

Notes
-----
All public functions validate their inputs and raise :exc:`ValueError`
for mismatched lengths or empty arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _validate(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Cast inputs to float arrays, flatten, and validate shapes."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length. "
            f"Got {len(y_true)} vs {len(y_pred)}."
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
    y_pred : array-like, shape (n,)

    Returns
    -------
    float
        RMSE >= 0.
    """
    y_true, y_pred = _validate(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> float:
    """Normalised RMSE (range-normalised).

    Computed as ``rmse / (max(y_true) - min(y_true))``.
    Returns ``nan`` when all true values are identical (zero range).

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

    Returns
    -------
    float
        NRMSE >= 0, or ``nan`` for a constant series.
    """
    y_true, y_pred = _validate(y_true, y_pred)
    r = float(np.max(y_true) - np.min(y_true))
    if r == 0.0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / r)


def mape(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
    eps: float = 1e-10,
) -> float:
    """Mean Absolute Percentage Error.

    Samples where ``|y_true| < eps`` are excluded.  Returns ``nan`` when
    *all* true values are near zero.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)
    eps : float, default 1e-10
        Threshold below which a true value is considered zero.

    Returns
    -------
    float
        MAPE >= 0 (as a fraction, not a percentage), or ``nan``.
    """
    y_true, y_pred = _validate(y_true, y_pred)
    mask = np.abs(y_true) >= eps
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def smape(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> float:
    """Symmetric Mean Absolute Percentage Error.

    Computed as ``mean(|y_true - y_pred| / ((|y_true| + |y_pred|) / 2))``.
    Degenerate samples where both values are zero contribute 0.
    The result is bounded in ``[0, 2]``.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

    Returns
    -------
    float
        sMAPE in [0, 2].
    """
    y_true, y_pred = _validate(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    num = np.abs(y_true - y_pred)
    # avoid 0/0: when both values are zero the error is 0
    ratio = np.where(denom == 0.0, 0.0, num / denom)
    return float(np.mean(ratio))


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------


def compute_all(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> dict[str, float]:
    """Compute all four metrics and return them as a dictionary.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

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
