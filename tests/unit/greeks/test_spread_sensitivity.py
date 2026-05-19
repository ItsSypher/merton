"""Tests for the spread-sensitivity helper."""

from __future__ import annotations

import numpy as np
import pytest

from merton.exceptions import MertonInputError
from merton.greeks import spread_sensitivity

A = np.array([100.0, 150.0, 200.0])
S = np.array([0.25, 0.30, 0.35])
D = np.array([60.0, 80.0, 100.0])
R = 0.04
T = 1.0


class TestSpreadSensitivity:
    def test_leverage_sensitivity_positive(self) -> None:
        v = spread_sensitivity(A, S, D, R, T, wrt="leverage")
        assert np.all(v > 0)  # more debt ⇒ higher spread

    def test_vol_sensitivity_positive(self) -> None:
        v = spread_sensitivity(A, S, D, R, T, wrt="vol")
        # Higher asset vol increases PD which increases spread.
        assert np.all(v > 0)

    def test_rate_sensitivity_finite(self) -> None:
        v = spread_sensitivity(A, S, D, R, T, wrt="rate")
        assert np.all(np.isfinite(v))

    def test_unknown_wrt_raises(self) -> None:
        with pytest.raises(MertonInputError):
            spread_sensitivity(A, S, D, R, T, wrt="bogus")  # type: ignore[arg-type]

    def test_higher_lgd_increases_sensitivity_magnitude(self) -> None:
        v_low = spread_sensitivity(A, S, D, R, T, wrt="leverage", lgd=0.2)
        v_hi = spread_sensitivity(A, S, D, R, T, wrt="leverage", lgd=0.9)
        # Spread scales monotonically with LGD when PD * LGD is small.
        assert np.all(v_hi >= v_low)
