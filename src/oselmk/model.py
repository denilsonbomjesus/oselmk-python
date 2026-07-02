"""OS-ELMK: Online Sequential Extreme Learning Machine with Kernels.

This module implements the OS-ELMK algorithm proposed in:

    Huang, G., et al. (2014). "Online sequential extreme learning machine
    with kernels for nonstationary time series prediction."
    Neurocomputing, 145, 90-97.
    https://doi.org/10.1016/j.neucom.2014.05.068

Only regression is supported, matching the scope of the paper.

Bug fixed from the Octave reference implementation
--------------------------------------------------
The original ``os_elmk_model.m`` contained a dead-code block inside a
classification branch that referenced ``TV.T`` and
``NumberofTestingData`` -- variables that do not exist in that function
(they were copy-pasted verbatim from ``elm_kernel.m`` without adaptation).
That block would raise a runtime error if classification mode were ever
triggered.  This implementation removes it entirely and restricts the
interface to regression, which is the only mode validated in the paper.

Numerical stability
-------------------
The original code used matrix inversion::

    R_inv = inv(Omega + I/C)

This implementation uses ``scipy.linalg.solve(A, b, assume_a='pos')``
instead.  Solving a linear system is numerically more stable than
explicit inversion and roughly twice as fast for symmetric positive
definite matrices.

Prediction caching
------------------
The original ``predict_os_elmk.m`` recomputed
``OutputWeight = R_inv \\ T`` on every prediction call -- an O(n^2)
operation that is redundant when the model state has not changed.

Here, ``output_weight_`` is cached as a model attribute and recomputed
only when the internal state is stale, tracked by the boolean flag
``_weights_dirty``.  The flag is set to ``True`` by ``fit()`` and
``update()``; cleared after the first ``_get_output_weight()`` call.

Sequential update (incremental learning)
-----------------------------------------
Equations referenced below are from the paper (Section 2.3)::

    Eq. 17 : G      = -R_inv @ K_cross          (sensitivity)
    Eq. 20 : theta* = [theta + G @ theta_new,
                        theta_new]               (stacked multipliers)
    Eq. 26 : gamma  = K_new_new + I_bs/C
                      + K_cross.T @ G            (Schur complement)
    Eq. 27 : theta_new = gamma^{-1} @ E_bs       (new-block multipliers)

    Block matrix inverse (R_inv update, derived from Schur complement):

        R11 = R_inv + G @ gamma_inv @ G.T
        R12 = -G @ gamma_inv
        R21 = R12.T
        R22 = gamma_inv

        R_inv_new = [[R11, R12],
                     [R21, R22]]

Decremental update (sliding window)
------------------------------------
After the full (n + bs) incremental expansion described above, the
oldest ``bs`` support vectors are pruned by slicing every state matrix
from index ``bs`` onward::

    R_inv_  = R_inv_expanded[bs:, bs:]
    theta_  = theta_expanded[bs:]
    K_elm_  = K_elm_expanded[bs:, bs:]
    X_train_ = X_train_expanded[bs:]
    y_train_ = y_train_expanded[bs:]

This keeps ``n_train_`` constant at ``window_size`` and lets the model
forget old patterns, enabling adaptation to nonstationary series.

Note on the pruned R_inv sub-block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The (bs:, bs:) sub-block of the block-inverted R_inv is NOT the exact
inverse of the pruned kernel matrix ``(K_elm_[bs:,bs:] + I/C)``.  It is
an approximation that avoids an O(n^3) re-solve at every step.  The
reference Octave implementation makes the same approximation.  For
practical nonstationary forecasting the error is negligible as long as
the window is large enough relative to the block size.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from oselmk.utils.kernels import kernel_matrix


class OSELMK:
    """Online Sequential ELM with Kernels.

    Parameters
    ----------
    C : float, default=1.0
        Regularisation constant.  Larger values reduce regularisation.
        Must be strictly positive.
    kernel : {'rbf', 'linear', 'poly', 'wavelet'}, default='rbf'
        Kernel function to use.
    kernel_params : float or array-like, optional
        Parameters forwarded to the kernel function.
        * ``'rbf'``     : sigma (float, default 1.0)
        * ``'linear'``  : ignored
        * ``'poly'``    : [c, d]  (bias and degree, defaults [1.0, 2.0])
        * ``'wavelet'`` : [b, a, omega0] (defaults [1.0, 1.0, 1.0])

    Attributes
    ----------
    R_inv_ : ndarray, shape (n_support, n_support)
        Inverse of the regularised kernel matrix.  Grows with each
        sequential update; stays fixed size with decremental updates.
    theta_ : ndarray, shape (n_support, n_outputs)
        Lagrange multipliers.  Equivalent to ``R_inv_ @ y_train_``.
    output_weight_ : ndarray, shape (n_support, n_outputs) or None
        Cached output weight matrix.  ``None`` until the first
        ``predict()`` call after ``fit()`` or ``update()``.
    _weights_dirty : bool
        Cache invalidation flag.  Set by ``fit()`` and ``update()``.
    X_train_ : ndarray, shape (n_support, n_features)
        Support inputs.
    y_train_ : ndarray, shape (n_support, n_outputs)
        Support targets.
    K_elm_ : ndarray, shape (n_support, n_support)
        Kernel matrix (without regularisation).
    n_train_ : int
        Current number of support vectors.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: Literal["rbf", "linear", "poly", "wavelet"] = "rbf",
        kernel_params: float | list[float] | None = None,
    ) -> None:
        if C <= 0:
            raise ValueError(f"C must be strictly positive, got {C}.")
        self.C = C
        self.kernel = kernel
        self.kernel_params = kernel_params

        self.R_inv_: NDArray[np.floating] | None = None
        self.theta_: NDArray[np.floating] | None = None
        self.X_train_: NDArray[np.floating] | None = None
        self.y_train_: NDArray[np.floating] | None = None
        self.K_elm_: NDArray[np.floating] | None = None
        self.n_train_: int | None = None

        self.output_weight_: NDArray[np.floating] | None = None
        self._weights_dirty: bool = True

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> "OSELMK":
        """Train the ELMK model on a batch of data (offline phase).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self
        """
        X, y = self._validate_Xy(X, y)
        n = X.shape[0]

        K = kernel_matrix(X, X, kernel=self.kernel, params=self.kernel_params)
        A = K + np.eye(n) / self.C
        theta = linalg.solve(A, y, assume_a="pos")
        R_inv = linalg.solve(A, np.eye(n), assume_a="pos")

        self.K_elm_ = K
        self.R_inv_ = R_inv
        self.theta_ = theta
        self.X_train_ = X
        self.y_train_ = y
        self.n_train_ = n
        self.output_weight_ = None
        self._weights_dirty = True
        return self

    # ------------------------------------------------------------------
    # Online update  (sequential + decremental)
    # ------------------------------------------------------------------

    def update(
        self,
        X_new: NDArray[np.floating],
        y_new: NDArray[np.floating],
        mode: Literal["sequential", "decremental"] = "sequential",
        window_size: int | None = None,
    ) -> "OSELMK":
        """Update the model online with a new block of data.

        Two modes are supported:

        ``'sequential'`` (incremental)
            The support set **grows** by ``bs`` samples.  Implements
            Section 2.3 of the paper (Eqs. 17, 20, 26, 27) via the
            Schur-complement block-matrix inverse.  Cost: O(n·bs + bs³)
            instead of O((n+bs)³).

        ``'decremental'`` (sliding window)
            First performs the full sequential expansion to size
            ``(n + bs)``, then prunes the oldest ``bs`` rows/columns
            from every state matrix, keeping ``n_train_`` fixed at
            ``window_size``.  Equivalent to ``os_elmk_dec_learning.m``.
            Enables forgetting of old patterns for nonstationary series.

            ``window_size`` must equal ``n_train_`` (the current support
            size at call time).  If omitted it defaults to the current
            ``n_train_``.

        Equations (paper notation)
        --------------------------
        Let ``n`` = current support size, ``bs`` = block size.

        K_cross  : kernel(X_train_, X_new)          shape (n,  bs)
        K_new    : kernel(X_new, X_new)             shape (bs, bs)

        G        = -R_inv @ K_cross                 (Eq. 17)  (n,  bs)
        gamma    = K_new + I_bs/C + K_cross.T @ G   (Eq. 26)  (bs, bs)
        E_bs     = y_new - K_cross.T @ output_weight_          (bs, n_out)
        theta_new = gamma^{-1} @ E_bs               (Eq. 27)  (bs, n_out)

        # Stacked multipliers (Eq. 20)
        delta_theta = G @ theta_new                 (n,  n_out)
        theta*      = [theta + delta_theta;
                       theta_new]

        # Block-matrix inverse update (derived from Schur complement)
        gamma_inv = inv(gamma)                      (bs, bs)
        R11 = R_inv + G @ gamma_inv @ G.T           (n,  n)
        R12 = -G @ gamma_inv                        (n,  bs)
        R_inv_new = [[R11, R12],
                     [R12.T, gamma_inv]]            (n+bs, n+bs)

        Decremental pruning (applied only when mode='decremental')
        -----------------------------------------------------------
        After expansion to (n+bs), slice away the first ``bs`` indices::

            R_inv_  = R_inv_expanded[bs:, bs:]
            theta_  = theta_expanded[bs:]
            K_elm_  = K_elm_expanded[bs:, bs:]
            X_train_ = X_train_expanded[bs:]
            y_train_ = y_train_expanded[bs:]

        Parameters
        ----------
        X_new : array-like, shape (bs, n_features)
            New input block.
        y_new : array-like, shape (bs,) or (bs, n_outputs)
            New target block.
        mode : {'sequential', 'decremental'}, default 'sequential'
            Update mode.
        window_size : int or None, default None
            Only used when ``mode='decremental'``.  Must equal the
            current ``n_train_``.  Defaults to ``self.n_train_``.

        Returns
        -------
        self
        """
        self._check_is_fitted()

        if mode not in ("sequential", "decremental"):
            raise NotImplementedError(
                f"mode='{mode}' is not recognised. Choose 'sequential' or 'decremental'."
            )

        X_new, y_new = self._validate_Xy(X_new, y_new)
        bs = X_new.shape[0]

        if mode == "decremental":
            # Validate / default window_size
            if window_size is None:
                window_size = self.n_train_
            if window_size != self.n_train_:
                raise ValueError(
                    f"window_size ({window_size}) must equal the current "
                    f"n_train_ ({self.n_train_}).  The sliding window "
                    "keeps the support size fixed."
                )
            if bs > window_size:
                raise ValueError(
                    f"Block size bs={bs} exceeds window_size={window_size}. "
                    "bs must be strictly smaller than the window."
                )

        # --- Shared incremental expansion (sequential + decremental) ---

        R_inv = self.R_inv_  # (n, n)
        theta = self.theta_  # (n, n_out)
        X_old = self.X_train_  # (n, d)
        K_old = self.K_elm_  # (n, n)
        w = self._get_output_weight()  # (n, n_out) -- cached

        # K_cross[i,j] = k(x_old_i, x_new_j)      shape (n, bs)
        K_cross = kernel_matrix(X_old, X_new, kernel=self.kernel, params=self.kernel_params)
        # K_new[i,j]   = k(x_new_i, x_new_j)      shape (bs, bs)
        K_new = kernel_matrix(X_new, X_new, kernel=self.kernel, params=self.kernel_params)

        # Eq. 17 -- sensitivity matrix             shape (n, bs)
        G = -(R_inv @ K_cross)

        # Eq. 26 -- Schur complement (gamma)        shape (bs, bs)
        gamma = K_new + np.eye(bs) / self.C + K_cross.T @ G

        # gamma is SPD; solve once, reuse for theta_new and R_inv update
        gamma_inv = linalg.solve(gamma, np.eye(bs), assume_a="pos")

        # Eq. 27 -- new-block Lagrange multipliers  shape (bs, n_out)
        E_bs = y_new - K_cross.T @ w
        theta_new = gamma_inv @ E_bs

        # Eq. 20 -- stacked multipliers             shape (n+bs, n_out)
        delta_theta = G @ theta_new
        theta_star = np.vstack([theta + delta_theta, theta_new])

        # Block-matrix inverse update               shape (n+bs, n+bs)
        R11 = R_inv + G @ gamma_inv @ G.T  # (n,  n)
        R12 = G @ gamma_inv  # (n,  bs)
        R_inv_expanded = np.block([[R11, R12], [R12.T, gamma_inv]])

        # Expanded kernel matrix                    shape (n+bs, n+bs)
        K_expanded = np.block(
            [
                [K_old, K_cross],
                [K_cross.T, K_new],
            ]
        )

        X_expanded = np.vstack([X_old, X_new])
        y_expanded = np.vstack([self.y_train_, y_new])

        # --- Mode-specific finalisation --------------------------------

        if mode == "sequential":
            self.R_inv_ = R_inv_expanded
            self.theta_ = theta_star
            self.K_elm_ = K_expanded
            self.X_train_ = X_expanded
            self.y_train_ = y_expanded
            self.n_train_ = self.n_train_ + bs

        else:  # mode == "decremental"
            # Prune the oldest `bs` support vectors (sliding window).
            # Equivalent to os_elmk_dec_learning.m lines:
            #   model.theta = theta_ast((bs+1):end)
            #   model.K_elm = K_elm_expanded((bs+1):end, (bs+1):end)
            #   model.R_inv = R_inv_expanded((bs+1):end, (bs+1):end)
            self.R_inv_ = R_inv_expanded[bs:, bs:]
            self.theta_ = theta_star[bs:]
            self.K_elm_ = K_expanded[bs:, bs:]
            self.X_train_ = X_expanded[bs:]
            self.y_train_ = y_expanded[bs:]
            # n_train_ stays fixed (window_size)
            # self.n_train_ is unchanged

        # Invalidate prediction cache
        self.output_weight_ = None
        self._weights_dirty = True
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        X: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Predict using the current model state.

        Uses the cached ``output_weight_`` (``R_inv_ @ y_train_``) and
        recomputes only when the state is stale.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray, shape (n_samples,) for single-output.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D array, got shape {X.shape}.")

        K_test = kernel_matrix(X, self.X_train_, kernel=self.kernel, params=self.kernel_params)
        return (K_test @ self._get_output_weight()).squeeze()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _get_output_weight(self) -> NDArray[np.floating]:
        """Return ``R_inv_ @ y_train_``, computing only when stale."""
        if self._weights_dirty or self.output_weight_ is None:
            self.output_weight_ = self.R_inv_ @ self.y_train_
            self._weights_dirty = False
        return self.output_weight_

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_Xy(
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Coerce and validate (X, y) inputs."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2-D array of shape (n_samples, n_features), got shape {X.shape}."
            )
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.ndim != 2:
            raise ValueError(f"y must be 1-D or 2-D, got shape {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}."
            )
        return X, y

    def _check_is_fitted(self) -> None:
        """Raise RuntimeError if the model has not been fitted yet."""
        if self.R_inv_ is None:
            raise RuntimeError(
                "This OSELMK instance is not fitted yet. "
                "Call fit(X, y) before calling predict or update."
            )
