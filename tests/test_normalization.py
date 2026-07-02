"""Unit tests for src/oselmk/utils/normalization.py."""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from oselmk.utils.normalization import ZScoreNormalizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

def test_fit_returns_self():
    z = ZScoreNormalizer()
    assert z.fit(RNG.standard_normal((10, 3))) is z

def test_fit_stores_mean_and_std():
    X = RNG.standard_normal((50, 4))
    z = ZScoreNormalizer().fit(X)
    assert z.mean_.shape == (4,)
    assert z.std_.shape == (4,)

def test_fit_mean_correct():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    z = ZScoreNormalizer().fit(X)
    assert_allclose(z.mean_, [3.0, 4.0])

def test_fit_std_correct():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    z = ZScoreNormalizer().fit(X)
    assert_allclose(z.std_, np.std(X, axis=0, ddof=0))

def test_fit_raises_on_1d_input():
    with pytest.raises(ValueError, match="2-D"):
        ZScoreNormalizer().fit(np.ones(5))

def test_fit_raises_on_empty_input():
    with pytest.raises(ValueError, match="at least one sample"):
        ZScoreNormalizer().fit(np.ones((0, 3)))


# ---------------------------------------------------------------------------
# transform()
# ---------------------------------------------------------------------------

def test_transform_zero_mean():
    X = RNG.standard_normal((30, 3))
    Xn = ZScoreNormalizer().fit(X).transform(X)
    assert_allclose(Xn.mean(axis=0), np.zeros(3), atol=1e-12)

def test_transform_unit_std():
    X = RNG.standard_normal((30, 3))
    Xn = ZScoreNormalizer().fit(X).transform(X)
    assert_allclose(Xn.std(axis=0), np.ones(3), atol=1e-12)

def test_transform_constant_column_becomes_zero():
    X = np.ones((5, 2))
    Xn = ZScoreNormalizer().fit(X).transform(X)
    assert_allclose(Xn, np.zeros((5, 2)))

def test_transform_constant_column_warns():
    X = np.ones((5, 2))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ZScoreNormalizer().fit(X).transform(X)
    assert any("constant" in str(warning.message).lower() for warning in w)

def test_transform_raises_before_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        ZScoreNormalizer().transform(np.ones((5, 3)))

def test_transform_raises_on_feature_mismatch():
    z = ZScoreNormalizer().fit(np.ones((5, 3)))
    with pytest.raises(ValueError, match="features"):
        z.transform(np.ones((5, 4)))


# ---------------------------------------------------------------------------
# inverse_transform()
# ---------------------------------------------------------------------------

def test_inverse_transform_recovers_original():
    X = RNG.standard_normal((20, 3))
    z = ZScoreNormalizer().fit(X)
    assert_allclose(z.inverse_transform(z.transform(X)), X, atol=1e-12)

def test_inverse_transform_raises_before_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        ZScoreNormalizer().inverse_transform(np.ones((5, 3)))

def test_inverse_transform_raises_on_feature_mismatch():
    z = ZScoreNormalizer().fit(np.ones((5, 3)))
    with pytest.raises(ValueError, match="features"):
        z.inverse_transform(np.ones((5, 4)))


# ---------------------------------------------------------------------------
# fit_transform()
# ---------------------------------------------------------------------------

def test_fit_transform_equals_fit_then_transform():
    X = RNG.standard_normal((15, 2))
    z1 = ZScoreNormalizer().fit(X)
    Xn1 = z1.transform(X)
    Xn2 = ZScoreNormalizer().fit_transform(X)
    assert_allclose(Xn1, Xn2)
