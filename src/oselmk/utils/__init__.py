"""oselmk.utils — Utility helpers for the OS-ELMK pipeline.

This sub-package re-exports every public utility symbol so callers can
use short, flat imports::

    from oselmk.utils import (
        make_lag_features,   # windowing
        ZScoreNormalizer,    # normalisation
        compute_all,         # metrics
        kernel_matrix,       # kernel dispatcher
        save_run,            # results persistence
    )

All symbols are also importable from their originating sub-modules when
the specific module is preferred for clarity::

    from oselmk.utils.windowing    import make_lag_features
    from oselmk.utils.normalization import ZScoreNormalizer
    from oselmk.utils.metrics      import rmse, nrmse, mape, smape, compute_all
    from oselmk.utils.kernels      import kernel_matrix
    from oselmk.utils.results      import save_run

Public symbols
--------------

Windowing
~~~~~~~~~
:func:`make_lag_features`
    Convert a 1-D time series into a supervised (X, y) dataset by
    building lag features.  Equivalent to ``data_to_legs.m``.

Normalisation
~~~~~~~~~~~~~
:class:`ZScoreNormalizer`
    sklearn-compatible Z-score scaler with ``ddof=1`` (matches Octave's
    ``std()`` default).  Implements ``fit / transform / inverse_transform
    / fit_transform``.

Metrics
~~~~~~~
:func:`rmse`
    Root Mean Squared Error.
:func:`nrmse`
    Normalised RMSE (range-normalised; returns ``nan`` for constant series).
:func:`mape`
    Mean Absolute Percentage Error (excludes near-zero true values).
:func:`smape`
    Symmetric MAPE, bounded in [0, 2].
:func:`compute_all`
    Convenience aggregator: returns ``{'rmse', 'nrmse', 'mape', 'smape'}``
    in one call.  Used by all example scripts and :func:`save_run`.

Kernels
~~~~~~~
:func:`kernel_matrix`
    Unified dispatcher: computes a kernel matrix given two data matrices
    and a kernel name (``'rbf'``, ``'linear'``, ``'poly'``, ``'wavelet'``).
:func:`rbf_kernel`
    RBF / Gaussian kernel: ``exp(-||x-y||² / sigma)``.
:func:`linear_kernel`
    Linear kernel: ``x^T y``.
:func:`poly_kernel`
    Polynomial kernel: ``(x^T y + c)^d``.
:func:`wavelet_kernel`
    Morlet wavelet kernel: ``cos(ω₀‖x-y‖/a) · exp(-‖x-y‖²/b)``.

Results
~~~~~~~
:func:`save_run`
    Persist ``metrics.json``, ``config.json`` and ``predictions.csv`` for
    one experiment run into a timestamped sub-directory of ``results/``.
"""

from __future__ import annotations

from oselmk.utils.kernels import (
    kernel_matrix,
    linear_kernel,
    poly_kernel,
    rbf_kernel,
    wavelet_kernel,
)
from oselmk.utils.metrics import compute_all, mape, nrmse, rmse, smape
from oselmk.utils.normalization import ZScoreNormalizer
from oselmk.utils.results import save_run
from oselmk.utils.windowing import make_lag_features

__all__: list[str] = [
    # windowing
    "make_lag_features",
    # normalisation
    "ZScoreNormalizer",
    # metrics
    "rmse",
    "nrmse",
    "mape",
    "smape",
    "compute_all",
    # kernels
    "kernel_matrix",
    "rbf_kernel",
    "linear_kernel",
    "poly_kernel",
    "wavelet_kernel",
    # results
    "save_run",
]
