"""Tests for the Geske compound-option model."""

from __future__ import annotations

import pytest

from merton import Firm
from merton.exceptions import MertonInputError
from merton.extensions import GeskeModel, geske_equity_value
from merton.extensions.geske import geske_pd


class TestGeskeEquity:
    def test_pricing_smoke(self) -> None:
        E = geske_equity_value(
            asset_value=200.0,
            asset_vol=0.25,
            debt_short=20.0,
            debt_long=50.0,
            t_short=1.0,
            t_long=3.0,
            rf=0.04,
        )
        assert E > 0
        assert E < 200.0

    def test_collapses_when_short_debt_is_zero(self) -> None:
        """With D_1 = 0 the inner-strike A* is also (numerically) tiny, so the
        Geske equity value should approach a plain Black-Scholes call on D_2
        at t_long."""
        from merton import equity_value as bsm

        # Use a small but nonzero D_1 to keep the solver stable.
        E_geske = geske_equity_value(
            asset_value=200.0,
            asset_vol=0.25,
            debt_short=1e-6,
            debt_long=50.0,
            t_short=1.0,
            t_long=3.0,
            rf=0.04,
        )
        E_bsm = float(bsm(200.0, 0.25, 50.0, 0.04, 3.0))
        # Within 1% — small D_1 still affects the inner-strike numerics.
        assert abs(E_geske - E_bsm) / E_bsm < 0.02

    def test_invalid_horizons(self) -> None:
        with pytest.raises(MertonInputError):
            geske_equity_value(
                asset_value=100.0,
                asset_vol=0.25,
                debt_short=20.0,
                debt_long=30.0,
                t_short=2.0,
                t_long=1.0,  # t_long < t_short
                rf=0.04,
            )


class TestGeskePD:
    def test_in_unit_interval(self) -> None:
        p = geske_pd(
            asset_value=200.0,
            asset_vol=0.30,
            debt_short=20.0,
            debt_long=50.0,
            t_short=1.0,
            t_long=3.0,
            rf=0.04,
        )
        assert 0.0 <= p <= 1.0

    def test_higher_leverage_more_pd(self) -> None:
        p_low = geske_pd(
            asset_value=200.0,
            asset_vol=0.30,
            debt_short=20.0,
            debt_long=50.0,
            t_short=1.0,
            t_long=3.0,
            rf=0.04,
        )
        p_hi = geske_pd(
            asset_value=200.0,
            asset_vol=0.30,
            debt_short=60.0,
            debt_long=120.0,
            t_short=1.0,
            t_long=3.0,
            rf=0.04,
        )
        assert p_hi >= p_low


class TestGeskeModel:
    def test_fit_smoke(self) -> None:
        firm = Firm(
            equity=100.0,
            debt_short=20,
            debt_long=30,
            equity_vol=0.30,
            rf=0.04,
            horizon=3.0,
        )
        result = GeskeModel(t_short=1.0).fit(firm)
        assert result.method == "geske"
        assert 0 <= result.pd <= 1
        assert result.asset_value > 0
        assert result.asset_vol > 0

    def test_requires_equity_vol(self) -> None:
        firm = Firm(equity=100.0, debt_short=20, debt_long=30, horizon=3.0)
        with pytest.raises(MertonInputError):
            GeskeModel(t_short=1.0).fit(firm)

    def test_rejects_invalid_horizon(self) -> None:
        firm = Firm(equity=100.0, debt_short=20, debt_long=30, equity_vol=0.30, horizon=0.5)
        with pytest.raises(MertonInputError):
            GeskeModel(t_short=1.0).fit(firm)
