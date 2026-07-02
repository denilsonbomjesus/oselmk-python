"""Unit tests for src/oselmk/model.py (offline batch phase + prediction cache)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from oselmk.model import OSELMK

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

# Small regression dataset: y = 2*x0 + 3*x1
N_TRAIN = 30
N_FEATURES = 3
X_TRAIN = RNG.uniform(-1.0, 1.0, (N_TRAIN, N_FEATURES))
Y_TRAIN = 2.0 * X_TRAIN[:, 0] + 3.0 * X_TRAIN[:, 1]  # 1-D
Y_TRAIN_2D = Y_TRAIN.reshape(-1, 1)                   # 2-D single output

X_TEST = RNG.uniform(-1.0, 1.0, (10, N_FEATURES))

# A second, disjoint dataset to test cache invalidation on re-fit
X_TRAIN_B = RNG.uniform(-1.0, 1.0, (N_TRAIN, N_FEATURES))
Y_TRAIN_B = 5.0 * X_TRAIN_B[:, 0] - X_TRAIN_B[:, 2]


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_default_instantiation():
    """OSELMK instantiates with default parameters without error."""
    model = OSELMK()
    assert model.C == 1.0
    assert model.kernel == "rbf"
    assert model.R_inv_ is None


def test_initial_weights_dirty_flag():
    """_weights_dirty must be True before fit() is called."""
    model = OSELMK()
    assert model._weights_dirty is True


def test_initial_output_weight_is_none():
    """output_weight_ must be None before any predict() call."""
    model = OSELMK()
    assert model.output_weight_ is None


def test_invalid_C_raises():
    """Non-positive C must raise ValueError."""
    with pytest.raises(ValueError, match="strictly positive"):
        OSELMK(C=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        OSELMK(C=-5.0)


# ---------------------------------------------------------------------------
# fit() — basic execution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["rbf", "linear", "poly", "wavelet"])
def test_fit_runs_for_all_kernels(kernel):
    """fit() must complete without error for every supported kernel."""
    model = OSELMK(kernel=kernel, C=1.0)
    model.fit(X_TRAIN, Y_TRAIN)
    assert model.R_inv_ is not None


def test_fit_sets_weights_dirty():
    """fit() must set _weights_dirty=True and clear output_weight_."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert model._weights_dirty is True
    assert model.output_weight_ is None


def test_fit_accepts_1d_y():
    """fit() must accept 1-D y and reshape it internally."""
    model = OSELMK()
    model.fit(X_TRAIN, Y_TRAIN)  # 1-D
    assert model.y_train_.ndim == 2
    assert model.y_train_.shape == (N_TRAIN, 1)


def test_fit_accepts_2d_y():
    """fit() must accept 2-D y without modification."""
    model = OSELMK()
    model.fit(X_TRAIN, Y_TRAIN_2D)
    assert model.y_train_.shape == (N_TRAIN, 1)


def test_fit_returns_self():
    """fit() must return the estimator instance (sklearn convention)."""
    model = OSELMK()
    result = model.fit(X_TRAIN, Y_TRAIN)
    assert result is model


# ---------------------------------------------------------------------------
# fit() — fitted attribute shapes
# ---------------------------------------------------------------------------


def test_R_inv_is_square():
    """R_inv_ must be a square matrix of shape (n_train, n_train)."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert model.R_inv_.shape == (N_TRAIN, N_TRAIN)


def test_theta_shape_single_output():
    """theta_ must have shape (n_train, 1) for single-output regression."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert model.theta_.shape == (N_TRAIN, 1)


def test_K_elm_shape():
    """K_elm_ must be square with shape (n_train, n_train)."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert model.K_elm_.shape == (N_TRAIN, N_TRAIN)


def test_X_train_stored_correctly():
    """X_train_ must be identical to the training input."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert_allclose(model.X_train_, X_TRAIN)


def test_n_train_stored_correctly():
    """n_train_ must equal the number of training samples."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert model.n_train_ == N_TRAIN


# ---------------------------------------------------------------------------
# fit() — R_inv_ mathematical properties
# ---------------------------------------------------------------------------


def test_R_inv_is_symmetric():
    """R_inv_ must be symmetric (A is SPD => A^{-1} is SPD)."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert_allclose(model.R_inv_, model.R_inv_.T, atol=1e-10)


def test_R_inv_times_A_is_identity():
    """R_inv_ @ (K_elm_ + I/C) must be close to the identity matrix."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    A = model.K_elm_ + np.eye(N_TRAIN) / model.C
    product = model.R_inv_ @ A
    assert_allclose(product, np.eye(N_TRAIN), atol=1e-10)


# ---------------------------------------------------------------------------
# fit() — input validation
# ---------------------------------------------------------------------------


def test_fit_raises_for_1d_X():
    """fit() must raise ValueError when X is 1-D."""
    with pytest.raises(ValueError, match="2-D"):
        OSELMK().fit(X_TRAIN[:, 0], Y_TRAIN)  # 1-D X


def test_fit_raises_for_shape_mismatch():
    """fit() must raise ValueError when X and y have different n_samples."""
    with pytest.raises(ValueError, match="same number of samples"):
        OSELMK().fit(X_TRAIN, Y_TRAIN[:10])


# ---------------------------------------------------------------------------
# predict() — basic execution
# ---------------------------------------------------------------------------


def test_predict_runs_after_fit():
    """predict() must return an array after fit() without error."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    y_pred = model.predict(X_TEST)
    assert isinstance(y_pred, np.ndarray)


def test_predict_output_shape():
    """predict() must return shape (n_test,) for single-output regression."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    y_pred = model.predict(X_TEST)
    assert y_pred.shape == (X_TEST.shape[0],)


def test_predict_perfect_on_linear_kernel_low_noise():
    """With linear kernel and high C, training error must be near zero."""
    model = OSELMK(kernel="linear", C=1e6).fit(X_TRAIN, Y_TRAIN)
    y_pred_train = model.predict(X_TRAIN)
    residuals = np.abs(y_pred_train - Y_TRAIN)
    assert residuals.max() < 1.0


def test_predict_raises_before_fit():
    """predict() must raise RuntimeError if called before fit()."""
    with pytest.raises(RuntimeError, match="not fitted"):
        OSELMK().predict(X_TEST)


def test_predict_raises_for_1d_X():
    """predict() must raise ValueError when X is 1-D."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    with pytest.raises(ValueError, match="2-D"):
        model.predict(X_TEST[:, 0])


# ---------------------------------------------------------------------------
# predict() — output weight cache
# ---------------------------------------------------------------------------


def test_output_weight_none_before_predict():
    """output_weight_ must remain None until the first predict() call."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert model.output_weight_ is None


def test_output_weight_populated_after_predict():
    """output_weight_ must be set after the first predict() call."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.predict(X_TEST)
    assert model.output_weight_ is not None


def test_weights_dirty_false_after_predict():
    """_weights_dirty must be False after predict() clears the cache."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.predict(X_TEST)
    assert model._weights_dirty is False


def test_output_weight_cached_same_object_on_repeated_predict():
    """Repeated predict() calls must return the same cached weight object."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.predict(X_TEST)  # populates cache
    w_first = model.output_weight_
    model.predict(X_TEST)  # must reuse cache
    w_second = model.output_weight_
    assert w_first is w_second  # same object in memory, not just equal values


def test_output_weight_invalidated_after_refit():
    """Re-fitting the model must invalidate the cache (output_weight_=None)."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.predict(X_TEST)  # populates cache
    assert model.output_weight_ is not None

    model.fit(X_TRAIN_B, Y_TRAIN_B)  # re-fit on different data
    assert model.output_weight_ is None
    assert model._weights_dirty is True


def test_output_weight_changes_after_refit():
    """Weight cached after re-fit must differ from the original weight."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.predict(X_TEST)
    w_before = model.output_weight_.copy()

    model.fit(X_TRAIN_B, Y_TRAIN_B)
    model.predict(X_TEST)  # recomputes with new data
    w_after = model.output_weight_

    assert not np.allclose(w_before, w_after)


def test_output_weight_shape():
    """output_weight_ must have shape (n_train, n_outputs) after predict()."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.predict(X_TEST)
    assert model.output_weight_.shape == (N_TRAIN, 1)


def test_predict_results_consistent_with_and_without_cache():
    """Predictions from cache and from direct theta must be numerically equal."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    y_cached = model.predict(X_TEST)  # uses cached output_weight_

    # Direct computation via theta_ (as in the Commit 6 implementation)
    K_test = __import__('oselmk.utils.kernels', fromlist=['kernel_matrix']).kernel_matrix(
        X_TEST, X_TRAIN, kernel='rbf'
    )
    y_direct = (K_test @ model.theta_).squeeze()

    assert_allclose(y_cached, y_direct, atol=1e-12)
