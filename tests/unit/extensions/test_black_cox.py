"""Tests for the Black-Cox first-passage model."""

from __future__ import annotations

import numpy as np
import pytest

from merton import Firm
from merton.exceptions import MertonInputError
from merton.extensions import BlackCoxModel, black_cox_pd
from merton.extensions.black_cox import black_cox_survival


class TestBlackCoxPD:
    def test_well_capitalised_firm_low_pd(self) -> None:
        pd_bc = float(black_cox_pd(200.0, 0.2, 50.0, 0.04, 1.0))
        assert 0 <= pd_bc < 0.01

    def test_leveraged_firm_high_pd(self) -> None:
        pd_bc = float(black_cox_pd(50.0, 0.5, 80.0, 0.04, 1.0))
        assert 0.1 < pd_bc <= 1.0

    def test_bc_pd_geq_merton_pd(self) -> None:
        """Black-Cox PD must dominate Merton PD (more default opportunities)."""
        from merton import distance_to_default, prob_of_default

        # A modestly leveraged firm.
        A, sigma, D, r, T = 100.0, 0.4, 80.0, 0.04, 1.0
        pd_bc = float(black_cox_pd(A, sigma, D, r, T))
        dd = float(distance_to_default(A, sigma, D, r, T))
        pd_m = float(prob_of_default(dd))
        assert pd_bc >= pd_m - 1e-10

    def test_pd_in_unit_interval(self) -> None:
        for A in (50.0, 100.0, 200.0):
            for sigma in (0.1, 0.3, 0.5):
                p = float(black_cox_pd(A, sigma, 80.0, 0.04, 1.0))
                assert 0.0 <= p <= 1.0

    def test_vectorized(self) -> None:
        out = black_cox_pd(np.array([100.0, 50.0]), 0.3, np.array([60.0, 60.0]), 0.04, 1.0)
        assert out.shape == (2,)

    def test_barrier_growth_rate(self) -> None:
        """Higher γ (growing barrier) should increase PD."""
        pd_flat = float(black_cox_pd(100.0, 0.4, 80.0, 0.04, 1.0, barrier_growth_rate=0.0))
        pd_grow = float(black_cox_pd(100.0, 0.4, 80.0, 0.04, 1.0, barrier_growth_rate=0.05))
        assert pd_grow >= pd_flat

    def test_survival_complement(self) -> None:
        p = float(black_cox_pd(100.0, 0.3, 80.0, 0.04, 1.0))
        s = float(black_cox_survival(100.0, 0.3, 80.0, 0.04, 1.0))
        assert s + p == pytest.approx(1.0, abs=1e-12)

    def test_invalid_inputs(self) -> None:
        with pytest.raises(MertonInputError):
            black_cox_pd(0.0, 0.3, 80.0, 0.04, 1.0)
        with pytest.raises(MertonInputError):
            black_cox_pd(100.0, 0.0, 80.0, 0.04, 1.0)
        with pytest.raises(MertonInputError):
            black_cox_pd(100.0, 0.3, 0.0, 0.04, 1.0)
        with pytest.raises(MertonInputError):
            black_cox_pd(100.0, 0.3, 80.0, 0.04, 0.0)


class TestBlackCoxModel:
    def test_fit_smoke(self) -> None:
        firm = Firm(
            equity=100.0, debt_short=20, debt_long=30, equity_vol=0.30, rf=0.04, horizon=1.0
        )
        result = BlackCoxModel().fit(firm)
        assert result.method == "black_cox"
        assert 0.0 <= result.pd <= 1.0
        assert result.asset_value > firm.equity
        assert result.asset_vol > 0

    def test_requires_equity_vol(self) -> None:
        firm = Firm(equity=100.0, debt_short=20, debt_long=30)
        with pytest.raises(MertonInputError):
            BlackCoxModel().fit(firm)

    def test_dd_inverse_of_pd(self) -> None:
        firm = Firm(equity=50.0, debt_short=30, debt_long=50, equity_vol=0.5, rf=0.04)
        result = BlackCoxModel().fit(firm)
        # PD = Φ(-DD) ⇒ DD = -Φ⁻¹(PD)
        from scipy.stats import norm

        assert result.dd == pytest.approx(-norm.ppf(result.pd), rel=1e-6)
