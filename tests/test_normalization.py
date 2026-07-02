"""Unit tests for src/oselmk/utils/normalization.py."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from oselmk.utils.normalization import ZScoreNormalizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
# 2-D matrix: 20 samples x 4 features, with varying scales
X_2D = RNG.standard_normal((20, 4)) * np.array([1.0, 10.0, 0.5, 100.0]) + np.array(
    [0.0, 5.0, -3.0, 50.0]
)
# 1-D series
X_1D = RNG.standard_normal(30) * 5.0 + 2.0


# ---------------------------------------------------------------------------
# fit / transform / inverse_transform
# ---------------------------------------------------------------------------


def test_fit_sets_attributes():
    """After fit, mean_ and std_ must have shape (n_features,)."""
    norm = ZScoreNormalizer()
    norm.fit(X_2D)
    assert norm.mean_.shape == (4,)
    assert norm.std_.shape == (4,)
    assert norm.n_features_in_ == 4


def test_transform_mean_zero():
    """Transformed training data must have (approximately) zero mean per feature."""
    norm = ZScoreNormalizer()
    X_norm = norm.fit_transform(X_2D)
    assert_allclose(X_norm.mean(axis=0), np.zeros(4), atol=1e-10)


def test_transform_std_one():
    """Transformed training data must have (approximately) unit std per feature.

    We use ddof=1 consistently, so the sample std of the normalised data is 1.
    """
    norm = ZScoreNormalizer()
    X_norm = norm.fit_transform(X_2D)
    assert_allclose(np.std(X_norm, axis=0, ddof=1), np.ones(4), atol=1e-10)


def test_roundtrip_2d():
    """inverse_transform(transform(X)) must recover the original X."""
    norm = ZScoreNormalizer()
    X_norm = norm.fit_transform(X_2D)
    X_recovered = norm.inverse_transform(X_norm)
    assert_allclose(X_recovered, X_2D, atol=1e-10)


def test_roundtrip_1d():
    """Roundtrip must work for 1-D inputs and preserve the 1-D shape."""
    norm = ZScoreNormalizer()
    X_norm = norm.fit_transform(X_1D)
    assert X_norm.ndim == 1, "Output should remain 1-D for 1-D input"
    X_recovered = norm.inverse_transform(X_norm)
    assert X_recovered.ndim == 1
    assert_allclose(X_recovered, X_1D, atol=1e-10)


def test_transform_new_data():
    """transform() applied to held-out data must use the *training* statistics."""
    norm = ZScoreNormalizer()
    norm.fit(X_2D)
    X_new = RNG.standard_normal((5, 4)) * np.array([1.0, 10.0, 0.5, 100.0])
    X_norm = norm.transform(X_new)
    # Should not be exactly mean-zero because stats come from training data
    # but inverse_transform must still recover X_new
    X_recovered = norm.inverse_transform(X_norm)
    assert_allclose(X_recovered, X_new, atol=1e-10)


# ---------------------------------------------------------------------------
# Constant feature (zero variance)
# ---------------------------------------------------------------------------


def test_constant_feature_warning():
    """A column with zero variance must trigger a UserWarning."""
    X_const = X_2D.copy()
    X_const[:, 2] = 7.0  # constant column
    norm = ZScoreNormalizer()
    with pytest.warns(UserWarning, match="zero variance"):
        norm.fit(X_const)


def test_constant_feature_not_scaled():
    """A constant feature must remain unchanged after normalization."""
    X_const = X_2D.copy()
    X_const[:, 1] = 3.14
    norm = ZScoreNormalizer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        X_norm = norm.fit_transform(X_const)
    # Constant feature centred at mean 3.14, std replaced by 1 -> values become 0
    assert_allclose(X_norm[:, 1], np.zeros(len(X_const)), atol=1e-10)


def test_constant_feature_roundtrip():
    """inverse_transform must recover the original constant column."""
    X_const = X_2D.copy()
    X_const[:, 0] = -1.0
    norm = ZScoreNormalizer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        X_norm = norm.fit_transform(X_const)
    X_recovered = norm.inverse_transform(X_norm)
    assert_allclose(X_recovered, X_const, atol=1e-10)


# ---------------------------------------------------------------------------
# ddof parameter
# ---------------------------------------------------------------------------


def test_ddof_zero_uses_population_std():
    """With ddof=0 the stored std_ must equal np.std(X, ddof=0)."""
    norm = ZScoreNormalizer(ddof=0)
    norm.fit(X_2D)
    expected_std = np.std(X_2D, axis=0, ddof=0)
    assert_allclose(norm.std_, expected_std, atol=1e-12)


def test_ddof_one_uses_sample_std():
    """With ddof=1 (default) the stored std_ must equal np.std(X, ddof=1)."""
    norm = ZScoreNormalizer()
    norm.fit(X_2D)
    expected_std = np.std(X_2D, axis=0, ddof=1)
    assert_allclose(norm.std_, expected_std, atol=1e-12)


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------


def test_transform_before_fit_raises():
    """Calling transform before fit must raise RuntimeError."""
    norm = ZScoreNormalizer()
    with pytest.raises(RuntimeError, match="not fitted"):
        norm.transform(X_2D)


def test_inverse_transform_before_fit_raises():
    """Calling inverse_transform before fit must raise RuntimeError."""
    norm = ZScoreNormalizer()
    with pytest.raises(RuntimeError, match="not fitted"):
        norm.inverse_transform(X_2D)


def test_wrong_feature_count_raises():
    """Passing data with wrong n_features to transform must raise ValueError."""
    norm = ZScoreNormalizer()
    norm.fit(X_2D)  # 4 features
    X_bad = RNG.standard_normal((5, 2))
    with pytest.raises(ValueError, match="Expected 4 features"):
        norm.transform(X_bad)


def test_3d_input_raises():
    """3-D arrays must raise ValueError."""
    norm = ZScoreNormalizer()
    with pytest.raises(ValueError, match="shape"):
        norm.fit(np.ones((3, 4, 5)))
