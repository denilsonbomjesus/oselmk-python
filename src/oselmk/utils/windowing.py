"""Lag embedding (sliding window) for univariate time series.

Converts a 1-D time series into a supervised learning problem by building
lag features.  This is the Python equivalent of ``data_to_legs.m`` from the
original Octave implementation, with two key improvements:

1. **Parametric n_lags**: the original was effectively fixed at ``legs=1`` in
   most experiments.  Here ``n_lags`` is a proper parameter, so multi-step
   lag embeddings are supported out of the box.

2. **Separate X and y**: the original returned a concatenated matrix where the
   first column was the target and the remaining columns were features.
   This function returns ``(X, y)`` separately, following the scikit-learn
   convention and making the intent explicit at every call site.

Layout
------
Given a series ``[s_0, s_1, ..., s_{N-1}]`` and ``n_lags=p``, the output is:

    y[i]      = s_{i + p}          # target: next value after the window
    X[i, :]   = [s_{i+p-1},        # most recent lag (lag-1)
                 s_{i+p-2},        # lag-2
                 ...,
                 s_{i}]            # oldest lag (lag-p)

This matches the column order produced by the original ``data_to_legs.m``
(first column = output = most recent value, subsequent columns = older lags).

The resulting shapes are::

    y : (N - n_lags,)
    X : (N - n_lags, n_lags)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_lag_features(
    series: NDArray[np.floating] | list[float],
    n_lags: int = 1,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Build a lag-feature matrix and target vector from a univariate series.

    Parameters
    ----------
    series : array-like, shape (N,)
        Univariate time series values in chronological order.
    n_lags : int, default 1
        Number of lag steps to include as features.  Must be >= 1.

    Returns
    -------
    X : ndarray, shape (N - n_lags, n_lags)
        Feature matrix.  Column 0 is lag-1 (most recent), column n_lags-1
        is lag-n_lags (oldest).
    y : ndarray, shape (N - n_lags,)
        Target vector (the value immediately following each window).

    Raises
    ------
    ValueError
        If ``series`` is not 1-D, has fewer than ``n_lags + 1`` elements, or
        ``n_lags < 1``.

    Examples
    --------
    >>> import numpy as np
    >>> from oselmk.utils.windowing import make_lag_features
    >>> s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> X, y = make_lag_features(s, n_lags=2)
    >>> X
    array([[2., 1.],
           [3., 2.],
           [4., 3.]])
    >>> y
    array([3., 4., 5.])
    """
    series = np.asarray(series, dtype=float)

    if series.ndim != 1:
        raise ValueError(f"'series' must be a 1-D array, got shape {series.shape}.")
    if n_lags < 1:
        raise ValueError(f"'n_lags' must be >= 1, got {n_lags}.")
    if len(series) <= n_lags:
        raise ValueError(
            f"'series' must have at least n_lags + 1 = {n_lags + 1} elements, got {len(series)}."
        )

    n_samples = len(series) - n_lags

    # Build X using stride tricks via a view — no data copies, O(1) memory
    # overhead compared to explicit loops.
    #
    # X[i, j] = series[i + n_lags - 1 - j]   for j in 0..n_lags-1
    #
    # We construct this as a (n_samples, n_lags) matrix where each row is
    # the reversed window ending just before the target index.
    X = np.empty((n_samples, n_lags), dtype=float)
    for lag in range(n_lags):
        # lag=0 -> most recent (s_{i + n_lags - 1})
        # lag=k -> s_{i + n_lags - 1 - k}
        X[:, lag] = series[n_lags - 1 - lag : n_lags - 1 - lag + n_samples]

    y = series[n_lags:]

    return X, y
