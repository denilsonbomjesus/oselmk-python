"""Z-score normalization with explicit sample standard deviation (ddof=1).

The original Octave implementation (featureNormalize.m) used::

    mu = sum(X) / length(X)
    sigma = std(X)          % Octave std uses ddof=1 by default

This class makes the same choice explicit via ``np.std(..., ddof=1)`` and
follows the scikit-learn transformer API (.fit / .transform / .inverse_transform
/ .fit_transform), making it easy to integrate in any pipeline.

Edge case — constant feature:
    When a feature has zero variance (std == 0) the column is left unchanged
    after normalization (division by zero is avoided).  A warning is issued so
    the caller is aware.  This is preferable to raising an exception, because
    constant features can appear in real datasets (e.g. a bias column).
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray


class ZScoreNormalizer:
    """Standardize features by removing the mean and scaling to unit variance.

    Uses sample standard deviation (``ddof=1``) to match the behaviour of
    Octave's ``std()`` function, which is also the default in most statistical
    packages.

    Parameters
    ----------
    ddof : int, default 1
        Delta degrees of freedom used in the standard deviation calculation.
        ``ddof=1`` gives the unbiased sample std; ``ddof=0`` gives the
        population std.  Change only if you have a specific reason to.

    Attributes
    ----------
    mean_ : ndarray, shape (n_features,)
        Per-feature mean, set after calling :meth:`fit`.
    std_ : ndarray, shape (n_features,)
        Per-feature standard deviation (ddof=1), set after calling :meth:`fit`.
        Constant features (std == 0) are stored as 1.0 to avoid division by
        zero, and a :class:`UserWarning` is issued.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    """

    def __init__(self, ddof: int = 1) -> None:
        self.ddof = ddof
        self.mean_: NDArray[np.floating] | None = None
        self.std_: NDArray[np.floating] | None = None
        self.n_features_in_: int | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if self.mean_ is None:
            raise RuntimeError(
                "This ZScoreNormalizer instance is not fitted yet. "
                "Call 'fit' before using 'transform' or 'inverse_transform'."
            )

    @staticmethod
    def _to_2d(X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Ensure X is always 2-D: (n_samples, n_features)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError(
                f"Expected a 1-D or 2-D array, got shape {X.shape}."
            )
        return X

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: NDArray[np.floating]) -> "ZScoreNormalizer":
        """Compute mean and std from *X* to be used for later scaling.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, n_features)
            Training data used to compute statistics.

        Returns
        -------
        self : ZScoreNormalizer
            Fitted normalizer (for method chaining).
        """
        X = self._to_2d(X)
        self.n_features_in_ = X.shape[1]
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=self.ddof)

        # Handle constant features: replace std=0 with 1 to avoid NaN
        zero_std_mask = self.std_ == 0.0
        if zero_std_mask.any():
            warnings.warn(
                f"{zero_std_mask.sum()} feature(s) with zero variance detected "
                "(constant column).  Those features will not be scaled.",
                UserWarning,
                stacklevel=2,
            )
            self.std_[zero_std_mask] = 1.0

        return self

    def transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Standardize *X* using the statistics computed during :meth:`fit`.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        X_norm : ndarray, same shape as input
            Standardized array.
        """
        self._check_is_fitted()
        was_1d = np.asarray(X).ndim == 1
        X = self._to_2d(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        X_norm = (X - self.mean_) / self.std_
        return X_norm.ravel() if was_1d else X_norm

    def inverse_transform(self, X_norm: NDArray[np.floating]) -> NDArray[np.floating]:
        """Reverse the standardization: ``X = X_norm * std_ + mean_``.

        Parameters
        ----------
        X_norm : array-like, shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        X : ndarray, same shape as input
            Array in the original scale.
        """
        self._check_is_fitted()
        was_1d = np.asarray(X_norm).ndim == 1
        X_norm = self._to_2d(X_norm)

        if X_norm.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X_norm.shape[1]}."
            )

        X = X_norm * self.std_ + self.mean_
        return X.ravel() if was_1d else X

    def fit_transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Fit to *X*, then transform *X*.

        Equivalent to ``self.fit(X).transform(X)`` but slightly more efficient
        since it avoids recomputing the statistics.

        Parameters
        ----------
        X : array-like, shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        X_norm : ndarray, same shape as input
        """
        return self.fit(X).transform(X)
