"""Tests for the BSM pricing helpers and equity Greeks."""

from __future__ import annotations

import numpy as np

from merton import equity_value
from merton.core.pricing import d1_d2
from merton.greeks import (
    equity_delta,
    equity_gamma,
    equity_rho,
    equity_theta,
    equity_vega,
)


class TestBSMPricing:
    def test_equity_value_positive(self) -> None:
        E = float(equity_value(100.0, 0.25, 60.0, 0.04, 1.0))
        assert E > 0
        # Equity must be at most asset_value
        assert E < 100.0

    def test_put_call_parity_like_identity(self) -> None:
        """A - E = D·e^(-rT)·Φ(d2) - A·(1-Φ(d1))·e^(-qT) (the residual debt PV)."""
        A, s, D, r, T = 100.0, 0.25, 60.0, 0.04, 1.0
        E = float(equity_value(A, s, D, r, T))
        d1, d2 = d1_d2(A, s, D, r, T)
        from scipy.special import ndtr

        expected_debt_value = D * np.exp(-r * T) * ndtr(d2) + A * (1 - ndtr(d1))
        np.testing.assert_allclose(A - E, expected_debt_value, rtol=1e-7)

    def test_higher_vol_higher_equity(self) -> None:
        """Equity (a call option) is monotone increasing in σ."""
        E_low = float(equity_value(100.0, 0.10, 60.0, 0.04, 1.0))
        E_high = float(equity_value(100.0, 0.40, 60.0, 0.04, 1.0))
        assert E_high > E_low


class TestEquityGreeks:
    def test_delta_in_unit_interval(self) -> None:
        delta = float(equity_delta(100.0, 0.25, 60.0, 0.04, 1.0))
        assert 0 < delta < 1

    def test_vega_positive(self) -> None:
        vega = float(equity_vega(100.0, 0.25, 60.0, 0.04, 1.0))
        assert vega > 0

    def test_gamma_positive(self) -> None:
        gamma = float(equity_gamma(100.0, 0.25, 60.0, 0.04, 1.0))
        assert gamma > 0

    def test_rho_positive_for_call(self) -> None:
        rho = float(equity_rho(100.0, 0.25, 60.0, 0.04, 1.0))
        assert rho > 0

    def test_theta_finite(self) -> None:
        theta = float(equity_theta(100.0, 0.25, 60.0, 0.04, 1.0))
        assert np.isfinite(theta)
