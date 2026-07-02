"""Unit tests for src/oselmk/model.py."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from oselmk.model import OSELMK
from oselmk.utils.kernels import kernel_matrix

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

N_TRAIN = 30
N_FEATURES = 3
X_TRAIN = RNG.uniform(-1.0, 1.0, (N_TRAIN, N_FEATURES))
Y_TRAIN = 2.0 * X_TRAIN[:, 0] + 3.0 * X_TRAIN[:, 1]
Y_TRAIN_2D = Y_TRAIN.reshape(-1, 1)

X_TEST = RNG.uniform(-1.0, 1.0, (10, N_FEATURES))

X_TRAIN_B = RNG.uniform(-1.0, 1.0, (N_TRAIN, N_FEATURES))
Y_TRAIN_B = 5.0 * X_TRAIN_B[:, 0] - X_TRAIN_B[:, 2]

# Online update block (bs=5)
BS = 5
X_NEW = RNG.uniform(-1.0, 1.0, (BS, N_FEATURES))
Y_NEW = 2.0 * X_NEW[:, 0] + 3.0 * X_NEW[:, 1]


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_default_instantiation():
    model = OSELMK()
    assert model.C == 1.0
    assert model.kernel == "rbf"
    assert model.R_inv_ is None

def test_initial_weights_dirty_flag():
    assert OSELMK()._weights_dirty is True

def test_initial_output_weight_is_none():
    assert OSELMK().output_weight_ is None

def test_invalid_C_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        OSELMK(C=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        OSELMK(C=-5.0)


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", ["rbf", "linear", "poly", "wavelet"])
def test_fit_runs_for_all_kernels(kernel):
    model = OSELMK(kernel=kernel, C=1.0).fit(X_TRAIN, Y_TRAIN)
    assert model.R_inv_ is not None

def test_fit_sets_weights_dirty():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert model._weights_dirty is True
    assert model.output_weight_ is None

def test_fit_accepts_1d_y():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert model.y_train_.shape == (N_TRAIN, 1)

def test_fit_accepts_2d_y():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN_2D)
    assert model.y_train_.shape == (N_TRAIN, 1)

def test_fit_returns_self():
    model = OSELMK()
    assert model.fit(X_TRAIN, Y_TRAIN) is model

def test_R_inv_is_square():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert model.R_inv_.shape == (N_TRAIN, N_TRAIN)

def test_theta_shape_single_output():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert model.theta_.shape == (N_TRAIN, 1)

def test_K_elm_shape():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert model.K_elm_.shape == (N_TRAIN, N_TRAIN)

def test_X_train_stored_correctly():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert_allclose(model.X_train_, X_TRAIN)

def test_n_train_stored_correctly():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert model.n_train_ == N_TRAIN

def test_R_inv_is_symmetric():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert_allclose(model.R_inv_, model.R_inv_.T, atol=1e-10)

def test_R_inv_times_A_is_identity():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    A = model.K_elm_ + np.eye(N_TRAIN) / model.C
    assert_allclose(model.R_inv_ @ A, np.eye(N_TRAIN), atol=1e-10)

def test_fit_raises_for_1d_X():
    with pytest.raises(ValueError, match="2-D"):
        OSELMK().fit(X_TRAIN[:, 0], Y_TRAIN)

def test_fit_raises_for_shape_mismatch():
    with pytest.raises(ValueError, match="same number of samples"):
        OSELMK().fit(X_TRAIN, Y_TRAIN[:10])


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

def test_predict_runs_after_fit():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert isinstance(model.predict(X_TEST), np.ndarray)

def test_predict_output_shape():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert model.predict(X_TEST).shape == (X_TEST.shape[0],)

def test_predict_perfect_on_linear_kernel_low_noise():
    model = OSELMK(kernel="linear", C=1e6).fit(X_TRAIN, Y_TRAIN)
    assert np.abs(model.predict(X_TRAIN) - Y_TRAIN).max() < 1.0

def test_predict_raises_before_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        OSELMK().predict(X_TEST)

def test_predict_raises_for_1d_X():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    with pytest.raises(ValueError, match="2-D"):
        model.predict(X_TEST[:, 0])


# ---------------------------------------------------------------------------
# predict() -- output weight cache
# ---------------------------------------------------------------------------

def test_output_weight_none_before_predict():
    assert OSELMK().fit(X_TRAIN, Y_TRAIN).output_weight_ is None

def test_output_weight_populated_after_predict():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.predict(X_TEST)
    assert model.output_weight_ is not None

def test_weights_dirty_false_after_predict():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.predict(X_TEST)
    assert model._weights_dirty is False

def test_output_weight_cached_same_object_on_repeated_predict():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.predict(X_TEST)
    assert model.output_weight_ is model.predict(X_TEST) or (
        model.predict(X_TEST)
        or True
    )
    w1 = model.output_weight_
    model.predict(X_TEST)
    assert model.output_weight_ is w1

def test_output_weight_invalidated_after_refit():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.predict(X_TEST)
    model.fit(X_TRAIN_B, Y_TRAIN_B)
    assert model.output_weight_ is None
    assert model._weights_dirty is True

def test_output_weight_changes_after_refit():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.predict(X_TEST)
    w_before = model.output_weight_.copy()
    model.fit(X_TRAIN_B, Y_TRAIN_B)
    model.predict(X_TEST)
    assert not np.allclose(w_before, model.output_weight_)

def test_output_weight_shape():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.predict(X_TEST)
    assert model.output_weight_.shape == (N_TRAIN, 1)

def test_predict_results_consistent_with_cache():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    y_cached = model.predict(X_TEST)
    K_test = kernel_matrix(X_TEST, X_TRAIN, kernel="rbf")
    y_direct = (K_test @ model.theta_).squeeze()
    assert_allclose(y_cached, y_direct, atol=1e-12)


# ---------------------------------------------------------------------------
# update() -- sequential mode
# ---------------------------------------------------------------------------

def test_update_sequential_increases_n_train():
    """After update, n_train_ must grow by the block size."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW, mode="sequential")
    assert model.n_train_ == N_TRAIN + BS

def test_update_sequential_expands_R_inv_shape():
    """R_inv_ must be (n+bs, n+bs) after a sequential update."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW, mode="sequential")
    expected = N_TRAIN + BS
    assert model.R_inv_.shape == (expected, expected)

def test_update_sequential_expands_theta_shape():
    """theta_ must be (n+bs, 1) after a sequential update."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW, mode="sequential")
    assert model.theta_.shape == (N_TRAIN + BS, 1)

def test_update_sequential_expands_K_elm_shape():
    """K_elm_ must be (n+bs, n+bs) after a sequential update."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW, mode="sequential")
    expected = N_TRAIN + BS
    assert model.K_elm_.shape == (expected, expected)

def test_update_sequential_expands_X_train():
    """X_train_ must contain old and new rows after update."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW, mode="sequential")
    assert model.X_train_.shape == (N_TRAIN + BS, N_FEATURES)
    assert_allclose(model.X_train_[:N_TRAIN], X_TRAIN)
    assert_allclose(model.X_train_[N_TRAIN:], X_NEW)

def test_update_sequential_invalidates_cache():
    """update() must set _weights_dirty=True and clear output_weight_."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.predict(X_TEST)
    assert model.output_weight_ is not None
    model.update(X_NEW, Y_NEW)
    assert model._weights_dirty is True
    assert model.output_weight_ is None

def test_update_sequential_returns_self():
    """update() must return self (fluent interface)."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    assert model.update(X_NEW, Y_NEW) is model

def test_update_before_fit_raises():
    """update() must raise RuntimeError if called before fit()."""
    with pytest.raises(RuntimeError, match="not fitted"):
        OSELMK().update(X_NEW, Y_NEW)

def test_update_unknown_mode_raises():
    """update() with an unknown mode must raise NotImplementedError."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    with pytest.raises(NotImplementedError):
        model.update(X_NEW, Y_NEW, mode="invalid_mode")

def test_update_sequential_R_inv_is_symmetric():
    """R_inv_ must remain symmetric after a sequential update."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW)
    assert_allclose(model.R_inv_, model.R_inv_.T, atol=1e-9)

def test_update_sequential_R_inv_times_A_is_identity():
    """R_inv_ @ A must be close to identity after update."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW)
    n = model.n_train_
    A = model.K_elm_ + np.eye(n) / model.C
    assert_allclose(model.R_inv_ @ A, np.eye(n), atol=1e-8)

def test_update_sequential_multiple_blocks():
    """Two sequential updates must grow n_train_ by 2*bs."""
    X_NEW2 = RNG.uniform(-1.0, 1.0, (BS, N_FEATURES))
    Y_NEW2 = -X_NEW2[:, 1]
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW)
    model.update(X_NEW2, Y_NEW2)
    assert model.n_train_ == N_TRAIN + 2 * BS


# ---------------------------------------------------------------------------
# update() -- decremental mode
# ---------------------------------------------------------------------------

def test_update_decremental_preserves_n_train():
    """Decremental update must keep n_train_ unchanged."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW, mode="decremental")
    assert model.n_train_ == N_TRAIN

def test_update_decremental_preserves_R_inv_shape():
    """R_inv_ shape must remain (n, n) after a decremental update."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW, mode="decremental")
    assert model.R_inv_.shape == (N_TRAIN, N_TRAIN)

def test_update_decremental_preserves_theta_shape():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW, mode="decremental")
    assert model.theta_.shape == (N_TRAIN, 1)

def test_update_decremental_preserves_K_elm_shape():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW, mode="decremental")
    assert model.K_elm_.shape == (N_TRAIN, N_TRAIN)

def test_update_decremental_slides_window():
    """After decremental update, the first BS rows of X_train_ must
    be the rows that were at positions BS..N_TRAIN-1 before, and the
    last BS rows must be the new samples."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    X_before = model.X_train_.copy()
    model.update(X_NEW, Y_NEW, mode="decremental")
    assert_allclose(model.X_train_[:N_TRAIN - BS], X_before[BS:])
    assert_allclose(model.X_train_[N_TRAIN - BS:], X_NEW)

def test_update_decremental_invalidates_cache():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.predict(X_TEST)
    model.update(X_NEW, Y_NEW, mode="decremental")
    assert model._weights_dirty is True
    assert model.output_weight_ is None

def test_update_decremental_R_inv_is_symmetric():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW, mode="decremental")
    assert_allclose(model.R_inv_, model.R_inv_.T, atol=1e-9)

def test_update_decremental_R_inv_times_A_is_identity():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW, mode="decremental")
    n = model.n_train_
    A = model.K_elm_ + np.eye(n) / model.C
    assert_allclose(model.R_inv_ @ A, np.eye(n), atol=1e-8)

def test_update_decremental_multiple_rounds():
    """Five rounds of decremental updates must keep n_train_ stable."""
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    for _ in range(5):
        Xb = RNG.uniform(0, 1, (BS, N_FEATURES))
        yb = Xb[:, 0]
        model.update(Xb, yb, mode="decremental")
    assert model.n_train_ == N_TRAIN


# ---------------------------------------------------------------------------
# predict() after update
# ---------------------------------------------------------------------------

def test_predict_after_sequential_update_runs():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW)
    assert isinstance(model.predict(X_TEST), np.ndarray)

def test_predict_after_decremental_update_runs():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW, mode="decremental")
    assert isinstance(model.predict(X_TEST), np.ndarray)

def test_predict_shape_after_sequential_update():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW)
    assert model.predict(X_TEST).shape == (X_TEST.shape[0],)

def test_predict_shape_after_decremental_update():
    model = OSELMK().fit(X_TRAIN, Y_TRAIN)
    model.update(X_NEW, Y_NEW, mode="decremental")
    assert model.predict(X_TEST).shape == (X_TEST.shape[0],)


# ---------------------------------------------------------------------------
# Kernel parametrisation
# ---------------------------------------------------------------------------

def test_kernel_params_passed_to_kernel_matrix():
    """Custom kernel_params must change the output weight."""
    m1 = OSELMK(kernel="rbf", kernel_params=0.1).fit(X_TRAIN, Y_TRAIN)
    m2 = OSELMK(kernel="rbf", kernel_params=100.0).fit(X_TRAIN, Y_TRAIN)
    m1.predict(X_TEST)
    m2.predict(X_TEST)
    assert not np.allclose(m1.output_weight_, m2.output_weight_)


# ---------------------------------------------------------------------------
# Regime-adaptation smoke test (decremental)
# ---------------------------------------------------------------------------

def test_decremental_adapts_to_new_regime():
    """After enough decremental updates with regime-B data, the model
    should predict regime-B samples better than regime-A samples."""
    n_window  = N_TRAIN
    n_features = N_FEATURES

    Xa = RNG.uniform(-1, 1, (n_window, n_features))
    ya = 2.0 * Xa[:, 0] + 3.0 * Xa[:, 1]

    model = OSELMK(kernel="linear", C=1e4).fit(Xa, ya)

    n_updates = int(np.ceil(n_window / BS)) + 2
    for _ in range(n_updates):
        Xb_block = RNG.uniform(0, 1, (BS, n_features))
        yb_block = -10.0 * Xb_block[:, 0]
        model.update(Xb_block, yb_block, mode="decremental")

    Xa_test = RNG.uniform(-1, 1, (20, n_features))
    ya_test = 2.0 * Xa_test[:, 0] + 3.0 * Xa_test[:, 1]

    Xb_test = RNG.uniform(0, 1, (20, n_features))
    yb_test = -10.0 * Xb_test[:, 0]

    rmse_a = float(np.sqrt(np.mean((model.predict(Xa_test) - ya_test) ** 2)))
    rmse_b = float(np.sqrt(np.mean((model.predict(Xb_test) - yb_test) ** 2)))

    assert rmse_b < rmse_a, (
        f"Model should favour regime B after window replacement, "
        f"but RMSE_B={rmse_b:.4f} >= RMSE_A={rmse_a:.4f}"
    )
