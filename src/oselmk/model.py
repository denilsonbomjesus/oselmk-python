"""OS-ELMK: Online Sequential Extreme Learning Machine with Kernels.

This module implements the offline batch phase (ELMK) of the OS-ELMK
algorithm proposed in:

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
    R_inv_ : ndarray, shape (n_train, n_train)
        Inverse of the regularised kernel matrix
        ``(Omega_train + I / C)``.  Stored for sequential updates.
    theta_ : ndarray, shape (n_train, n_outputs)
        Lagrange multipliers ``theta = C * (y - y_hat)``.  Used instead
        of explicit output weights; equivalent to
        ``OutputWeight = R_inv_ @ y_train_``.
    X_train_ : ndarray, shape (n_train, n_features)
        Training inputs retained for kernel evaluations during
        ``predict`` and online updates.
    y_train_ : ndarray, shape (n_train, n_outputs)
        Training targets retained for online updates.
    K_elm_ : ndarray, shape (n_train, n_train)
        Kernel matrix ``Omega_train`` (without regularisation).  Stored
        so that sequential updates can extend it without recomputation.
    n_train_ : int
        Number of training samples seen during ``fit``.
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

        # Fitted attributes -- set by fit()
        self.R_inv_: NDArray[np.floating] | None = None
        self.theta_: NDArray[np.floating] | None = None
        self.X_train_: NDArray[np.floating] | None = None
        self.y_train_: NDArray[np.floating] | None = None
        self.K_elm_: NDArray[np.floating] | None = None
        self.n_train_: int | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> "OSELMK":
        """Train the ELMK model on a batch of data (offline phase).

        Computes the regularised kernel matrix::

            A = Omega_train + I / C

        and solves the system to obtain the Lagrange multipliers::

            A @ theta = y_train

        ``scipy.linalg.solve`` with ``assume_a='pos'`` (Cholesky path) is
        used for numerical stability instead of explicit matrix inversion.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training inputs.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Training targets.

        Returns
        -------
        self : OSELMK
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2-D array of shape (n_samples, n_features), "
                f"got shape {X.shape}."
            )
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.ndim != 2:
            raise ValueError(
                f"y must be 1-D or 2-D, got shape {y.shape}."
            )
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got X: {X.shape[0]}, y: {y.shape[0]}."
            )

        n = X.shape[0]

        # --- kernel matrix (n x n) ----------------------------------------
        K = kernel_matrix(X, X, kernel=self.kernel, params=self.kernel_params)

        # --- regularised system matrix ------------------------------------
        # A = Omega + I/C  (Eq. 17 in the paper)
        A = K + np.eye(n) / self.C

        # --- solve A @ theta = y  (numerically stable, no explicit inv) ---
        # assume_a='pos': A is symmetric positive definite -> Cholesky path
        theta = linalg.solve(A, y, assume_a="pos")

        # --- store R_inv_ for online updates  (R_inv = A^{-1}) ------------
        # We need the explicit inverse for the rank-1 / rank-bs update
        # formulas (Eqs. 20, 26-27 in the paper).  We compute it by solving
        # A @ R_inv = I, reusing the same factorisation.
        R_inv = linalg.solve(A, np.eye(n), assume_a="pos")

        # --- store fitted attributes --------------------------------------
        self.K_elm_ = K
        self.R_inv_ = R_inv
        self.theta_ = theta
        self.X_train_ = X
        self.y_train_ = y
        self.n_train_ = n

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        X: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Predict using the fitted ELMK model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            Predicted values.  Single-output predictions are squeezed to 1-D.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2-D array, got shape {X.shape}."
            )

        # kernel between test samples and training samples
        K_test = kernel_matrix(
            X, self.X_train_, kernel=self.kernel, params=self.kernel_params
        )
        y_pred = K_test @ self.theta_
        return y_pred.squeeze()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        """Raise RuntimeError if the model has not been fitted yet."""
        if self.R_inv_ is None:
            raise RuntimeError(
                "This OSELMK instance is not fitted yet. "
                "Call fit(X, y) before calling predict or update."
            )
