"""Kernel functions for OS-ELMK.

Implements the four kernels from the reference paper:

    * RBF (Radial Basis Function / Gaussian)
    * Linear
    * Polynomial
    * Wavelet (Morlet-based)

All kernels are exposed through the unified ``kernel_matrix`` dispatcher,
which mirrors the interface of ``kernel_matrix_os.m`` from the Octave
reference implementation.

Design note
-----------
The Octave code computed full n×n kernel matrices using nested element-wise
operations.  Here, pairwise squared Euclidean distances are computed with
a single vectorised expression::

    ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j

This avoids explicit loops and is memory-efficient for large matrices.
"""

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: All kernel names accepted by :func:`kernel_matrix`.
SUPPORTED_KERNELS: tuple[str, ...] = ("rbf", "linear", "poly", "wavelet")


# ---------------------------------------------------------------------------
# Distance helper
# ---------------------------------------------------------------------------


def _pairwise_sq_dist(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute pairwise squared Euclidean distances between rows of A and B.

    Parameters
    ----------
    A : ndarray, shape (m, d)
    B : ndarray, shape (n, d)

    Returns
    -------
    D : ndarray, shape (m, n)
        D[i, j] = ||A[i] - B[j]||^2
    """
    sq_a = np.sum(A ** 2, axis=1, keepdims=True)  # (m, 1)
    sq_b = np.sum(B ** 2, axis=1, keepdims=True)  # (n, 1)
    dist_sq = sq_a + sq_b.T - 2.0 * (A @ B.T)
    return np.maximum(dist_sq, 0.0)


# ---------------------------------------------------------------------------
# Individual kernels
# ---------------------------------------------------------------------------


def rbf_kernel(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    sigma: float = 1.0,
) -> NDArray[np.floating]:
    """RBF (Gaussian) kernel: K(x,y) = exp(-||x-y||^2 / sigma).

    Parameters
    ----------
    A : ndarray, shape (m, d)
    B : ndarray, shape (n, d)
    sigma : float, default 1.0
        Bandwidth parameter.  Must be strictly positive.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be strictly positive, got {sigma}.")
    return np.exp(-_pairwise_sq_dist(A, B) / sigma)


def linear_kernel(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Linear kernel: K(x,y) = x^T y."""
    return A @ B.T


def poly_kernel(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    c: float = 1.0,
    d: float = 2.0,
) -> NDArray[np.floating]:
    """Polynomial kernel: K(x,y) = (x^T y + c)^d.

    Parameters
    ----------
    c : float, default 1.0
        Bias term.
    d : float, default 2.0
        Degree.
    """
    return (A @ B.T + c) ** d


def wavelet_kernel(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    b: float = 1.0,
    a: float = 1.0,
    omega0: float = 1.0,
) -> NDArray[np.floating]:
    """Wavelet (Morlet-based) kernel.

    K(x,y) = cos(omega0 * ||x-y|| / a) * exp(-||x-y||^2 / b)

    Parameters
    ----------
    b : float, default 1.0
        Envelope width (Gaussian decay).
    a : float, default 1.0
        Scale / dilation.
    omega0 : float, default 1.0
        Central frequency.
    """
    if a <= 0:
        raise ValueError(f"a must be strictly positive, got {a}.")
    if b <= 0:
        raise ValueError(f"b must be strictly positive, got {b}.")
    dist_sq = _pairwise_sq_dist(A, B)
    dist = np.sqrt(dist_sq)
    return np.cos(omega0 * dist / a) * np.exp(-dist_sq / b)


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def kernel_matrix(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    kernel: str = "rbf",
    params: float | list[float] | None = None,
) -> NDArray[np.floating]:
    """Compute a kernel matrix between rows of A and rows of B.

    Parameters
    ----------
    A : ndarray, shape (m, d)
    B : ndarray, shape (n, d)
    kernel : {'rbf', 'linear', 'poly', 'wavelet'}, default 'rbf'
    params : float or list of float, optional
        Kernel-specific parameters (see individual kernel functions).
        Defaults are applied when ``None``.

    Returns
    -------
    K : ndarray, shape (m, n)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)

    kernel = kernel.lower().strip()

    if kernel == "rbf":
        sigma = float(params) if params is not None else 1.0
        return rbf_kernel(A, B, sigma=sigma)

    if kernel == "linear":
        return linear_kernel(A, B)

    if kernel == "poly":
        if params is None:
            c, d = 1.0, 2.0
        else:
            p = list(params) if hasattr(params, "__iter__") else [params]
            if len(p) < 2:
                raise ValueError(
                    "Polynomial kernel requires two parameters: params=[c, d]."
                )
            c, d = float(p[0]), float(p[1])
        return poly_kernel(A, B, c=c, d=d)

    if kernel == "wavelet":
        if params is None:
            b, a, omega0 = 1.0, 1.0, 1.0
        else:
            p = list(params) if hasattr(params, "__iter__") else [params]
            if len(p) < 3:
                raise ValueError(
                    "Wavelet kernel requires three parameters: params=[b, a, w0]."
                )
            b, a, omega0 = float(p[0]), float(p[1]), float(p[2])
        return wavelet_kernel(A, B, b=b, a=a, omega0=omega0)

    raise ValueError(
        f"Unknown kernel '{kernel}'. "
        f"Choose from: {', '.join(repr(k) for k in SUPPORTED_KERNELS)}."
    )
