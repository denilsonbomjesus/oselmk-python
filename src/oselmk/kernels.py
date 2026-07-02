"""Kernel matrix computation for ELMK and OS-ELMK.

All functions follow the convention:
    X_train : ndarray of shape (n_train, n_features)
    X_test  : ndarray of shape (n_test,  n_features)  [optional]

When X_test is omitted the square training kernel Omega_train is returned.
When X_test is provided the rectangular cross-kernel Omega_test of shape
(n_train, n_test) is returned, so that predictions are computed as:
    y_hat = Omega_test.T @ output_weight

This is the opposite of the (n_features, n_samples) layout used by the
original Octave code, which required explicit transposes at every call site.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SUPPORTED_KERNELS = ("rbf", "linear", "polynomial", "wavelet")


def kernel_matrix(
    X_train: NDArray[np.floating],
    kernel_type: str,
    kernel_params: float | list[float],
    X_test: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Compute the kernel matrix between training (and optionally test) samples.

    Parameters
    ----------
    X_train : ndarray, shape (n_train, n_features)
        Training samples.
    kernel_type : str
        One of ``'rbf'``, ``'linear'``, ``'polynomial'``, ``'wavelet'``.
    kernel_params : float or list of float
        Kernel hyper-parameters:
        - ``'rbf'``        : sigma (scalar) — bandwidth
        - ``'linear'``     : ignored (pass any value, e.g. ``1.0``)
        - ``'polynomial'`` : [c, d] — (X @ X.T + c) ** d
        - ``'wavelet'``    : [b, a, w0] — b=scale, a=dilation, w0=central freq
    X_test : ndarray, shape (n_test, n_features), optional
        Test samples.  When omitted the square Omega_train is returned.

    Returns
    -------
    omega : ndarray
        Shape (n_train, n_train) when X_test is None,
        shape (n_train, n_test) otherwise.
    """
    kernel_type = kernel_type.lower()
    if kernel_type not in SUPPORTED_KERNELS:
        raise ValueError(
            f"Unknown kernel '{kernel_type}'. "
            f"Supported kernels: {SUPPORTED_KERNELS}"
        )

    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if X_test is not None:
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

    dispatch = {
        "rbf": _rbf_kernel,
        "linear": _linear_kernel,
        "polynomial": _polynomial_kernel,
        "wavelet": _wavelet_kernel,
    }
    return dispatch[kernel_type](X_train, kernel_params, X_test)


# ---------------------------------------------------------------------------
# Individual kernel implementations
# ---------------------------------------------------------------------------


def _squared_distances(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute pairwise squared Euclidean distances between rows of A and B.

    Uses the identity  ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b  which is
    fully vectorised and avoids explicit loops.

    Returns
    -------
    D : ndarray, shape (len(A), len(B))
    """
    # (n, 1) + (1, m) - 2*(n, m)  ->  (n, m)
    sq_A = np.sum(A ** 2, axis=1, keepdims=True)  # (n, 1)
    sq_B = np.sum(B ** 2, axis=1, keepdims=True)  # (m, 1)
    return sq_A + sq_B.T - 2.0 * (A @ B.T)


def _rbf_kernel(
    X_train: NDArray[np.floating],
    params: float | list[float],
    X_test: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
    """RBF (Gaussian) kernel: k(x, y) = exp(-||x - y||^2 / sigma)."""
    sigma = float(np.atleast_1d(params)[0])
    if sigma <= 0:
        raise ValueError(f"RBF kernel requires sigma > 0, got {sigma}.")

    B = X_train if X_test is None else X_test
    D = _squared_distances(X_train, B)
    return np.exp(-D / sigma)


def _linear_kernel(
    X_train: NDArray[np.floating],
    params: float | list[float],  # noqa: ARG001  (unused, kept for uniform signature)
    X_test: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
    """Linear kernel: k(x, y) = x.T y."""
    B = X_train if X_test is None else X_test
    return X_train @ B.T


def _polynomial_kernel(
    X_train: NDArray[np.floating],
    params: float | list[float],
    X_test: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
    """Polynomial kernel: k(x, y) = (x.T y + c) ** d."""
    p = np.atleast_1d(params)
    if len(p) < 2:
        raise ValueError(
            "Polynomial kernel requires params=[c, d], e.g. [0.0, 2].  "
            f"Got {params}."
        )
    c, d = float(p[0]), float(p[1])
    B = X_train if X_test is None else X_test
    return (X_train @ B.T + c) ** d


def _wavelet_kernel(
    X_train: NDArray[np.floating],
    params: float | list[float],
    X_test: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
    """Wavelet kernel: k(x, y) = cos(w0 * (x-y) / a) * exp(-||x-y||^2 / b).

    Parameters order: [b, a, w0]
        b  : controls the Gaussian envelope width
        a  : dilation factor
        w0 : central angular frequency
    """
    p = np.atleast_1d(params)
    if len(p) < 3:
        raise ValueError(
            "Wavelet kernel requires params=[b, a, w0].  "
            f"Got {params}."
        )
    b, a, w0 = float(p[0]), float(p[1]), float(p[2])

    B = X_train if X_test is None else X_test

    # Squared distances for the Gaussian envelope
    D_sq = _squared_distances(X_train, B)  # (n_train, n_B)

    # Sum-based differences for the cosine term
    # sum_A[i] - sum_B[j]  ->  (n_train, n_B)
    sum_A = X_train.sum(axis=1, keepdims=True)  # (n_train, 1)
    sum_B = B.sum(axis=1, keepdims=True)        # (n_B, 1)
    D_sum = sum_A - sum_B.T                     # (n_train, n_B)

    return np.cos(w0 * D_sum / a) * np.exp(-D_sq / b)
