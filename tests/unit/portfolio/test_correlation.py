"""Tests for the correlation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from merton.exceptions import MertonInputError
from merton.portfolio import asset_correlation_from_equity


class TestAssetCorrelation:
    def test_independent_returns_yield_identity_ish(self) -> None:
        rng = np.random.default_rng(0)
        R = rng.standard_normal(size=(2000, 5))
        corr = asset_correlation_from_equity(R)
        # Off-diagonals should be ≈ 0.
        off = corr - np.eye(5)
        assert np.max(np.abs(off)) < 0.06

    def test_with_leverage(self) -> None:
        rng = np.random.default_rng(0)
        R = rng.standard_normal(size=(500, 3))
        lev = np.array([0.5, 0.6, 0.4])
        corr = asset_correlation_from_equity(R, leverage=lev)
        # Diagonals must stay 1.
        np.testing.assert_allclose(np.diag(corr), 1.0)

    def test_shrinkage(self) -> None:
        rng = np.random.default_rng(0)
        R = rng.standard_normal(size=(500, 3))
        corr = asset_correlation_from_equity(R, shrinkage=1.0)
        np.testing.assert_allclose(corr, np.eye(3))

    def test_rejects_1d_returns(self) -> None:
        with pytest.raises(MertonInputError):
            asset_correlation_from_equity(np.zeros(100))

    def test_rejects_bad_shrinkage(self) -> None:
        with pytest.raises(MertonInputError):
            asset_correlation_from_equity(np.zeros((10, 2)), shrinkage=2.0)

    def test_rejects_wrong_leverage_shape(self) -> None:
        with pytest.raises(MertonInputError):
            asset_correlation_from_equity(np.zeros((10, 3)), leverage=np.array([0.5, 0.5]))
