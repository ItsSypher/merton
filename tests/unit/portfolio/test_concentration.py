"""Tests for HHI and granularity adjustment."""

from __future__ import annotations

import numpy as np
import pytest

from merton.exceptions import MertonInputError
from merton.portfolio import granularity_adjustment, hhi
from merton.portfolio.concentration import effective_n


class TestHHI:
    def test_perfectly_concentrated(self) -> None:
        assert hhi([1.0]) == 1.0

    def test_equal_exposures(self) -> None:
        for n in (2, 5, 100):
            assert hhi([1.0] * n) == pytest.approx(1.0 / n)

    def test_unbalanced(self) -> None:
        # 80/20 split: HHI = 0.64 + 0.04 = 0.68
        assert hhi([80, 20]) == pytest.approx(0.68)

    def test_effective_n(self) -> None:
        assert effective_n([1.0] * 5) == pytest.approx(5.0)

    def test_rejects_negative(self) -> None:
        with pytest.raises(MertonInputError):
            hhi([1.0, -1.0])

    def test_rejects_zero_total(self) -> None:
        with pytest.raises(MertonInputError):
            hhi([0.0, 0.0])

    def test_rejects_2d(self) -> None:
        with pytest.raises(MertonInputError):
            hhi(np.array([[1.0, 2.0]]))


class TestGranularityAdjustment:
    def test_returns_finite_positive(self) -> None:
        # For a small concentrated portfolio the GA correction must be > 0.
        ga = granularity_adjustment(
            pd=0.05, lgd=0.45, rho=0.20, exposures=np.array([10.0, 5.0, 3.0])
        )
        assert ga >= 0
        assert np.isfinite(ga)

    def test_rejects_2d_exposures(self) -> None:
        with pytest.raises(MertonInputError):
            granularity_adjustment(0.05, 0.45, 0.20, np.array([[1.0]]))
