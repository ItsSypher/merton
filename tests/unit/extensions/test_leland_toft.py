"""Tests for the Leland-Toft endogenous-default model."""

from __future__ import annotations

import numpy as np
import pytest

from merton import Firm
from merton.exceptions import MertonInputError
from merton.extensions import (
    LelandToftModel,
    leland_toft_debt_value,
    leland_toft_equity_value,
    leland_toft_pd,
    optimal_default_boundary,
)


class TestOptimalDefaultBoundary:
    def test_increases_with_coupon(self) -> None:
        v1 = float(optimal_default_boundary(coupon=1.0, rf=0.05, sigma_asset=0.3))
        v2 = float(optimal_default_boundary(coupon=10.0, rf=0.05, sigma_asset=0.3))
        assert v2 > v1

    def test_decreases_with_taxes(self) -> None:
        v_no_tax = float(
            optimal_default_boundary(coupon=5.0, rf=0.05, sigma_asset=0.3, tax_rate=0.0)
        )
        v_tax = float(optimal_default_boundary(coupon=5.0, rf=0.05, sigma_asset=0.3, tax_rate=0.4))
        assert v_tax < v_no_tax

    def test_rejects_invalid_rf(self) -> None:
        with pytest.raises(MertonInputError):
            optimal_default_boundary(coupon=5.0, rf=0.0, sigma_asset=0.3)

    def test_rejects_invalid_sigma(self) -> None:
        with pytest.raises(MertonInputError):
            optimal_default_boundary(coupon=5.0, rf=0.05, sigma_asset=0.0)

    def test_rejects_negative_coupon(self) -> None:
        with pytest.raises(MertonInputError):
            optimal_default_boundary(coupon=-1.0, rf=0.05, sigma_asset=0.3)

    def test_rejects_bad_tax_rate(self) -> None:
        with pytest.raises(MertonInputError):
            optimal_default_boundary(coupon=5.0, rf=0.05, sigma_asset=0.3, tax_rate=1.5)


class TestLelandToftPD:
    def test_pd_in_unit_interval(self) -> None:
        pd = float(
            leland_toft_pd(
                asset_value=100.0,
                asset_vol=0.30,
                coupon=5.0,
                rf=0.05,
            )
        )
        assert 0.0 <= pd <= 1.0

    def test_leveraged_higher_pd(self) -> None:
        low = float(leland_toft_pd(asset_value=100.0, asset_vol=0.30, coupon=2.0, rf=0.05))
        hi = float(leland_toft_pd(asset_value=100.0, asset_vol=0.30, coupon=8.0, rf=0.05))
        assert hi > low

    def test_below_barrier_returns_one(self) -> None:
        # Asset value below the optimal V_B ⇒ already in default.
        pd = float(
            leland_toft_pd(
                asset_value=5.0,
                asset_vol=0.30,
                coupon=10.0,
                rf=0.05,
                default_boundary=20.0,
            )
        )
        assert pd == 1.0

    def test_explicit_default_boundary(self) -> None:
        pd = float(
            leland_toft_pd(
                asset_value=100.0,
                asset_vol=0.30,
                coupon=5.0,
                rf=0.05,
                default_boundary=50.0,
            )
        )
        assert 0.0 < pd < 1.0

    def test_rejects_negative_asset(self) -> None:
        with pytest.raises(MertonInputError):
            leland_toft_pd(asset_value=-1.0, asset_vol=0.3, coupon=5.0, rf=0.05)

    def test_rejects_negative_default_boundary(self) -> None:
        with pytest.raises(MertonInputError):
            leland_toft_pd(
                asset_value=100.0, asset_vol=0.3, coupon=5.0, rf=0.05, default_boundary=-1.0
            )


class TestEquityAndDebtValues:
    def test_equity_plus_debt_equal_to_asset_minus_tax(self) -> None:
        """Equity + Debt should equal V_levered = V + tax shield - bankruptcy cost."""
        kw = {
            "asset_value": 100.0,
            "asset_vol": 0.30,
            "coupon": 5.0,
            "rf": 0.05,
            "tax_rate": 0.30,
            "bankruptcy_cost": 0.30,
        }
        E = float(leland_toft_equity_value(**kw))
        D = float(leland_toft_debt_value(**kw))
        # Tax shield: τ·C/r·(1 - PD); bankruptcy cost: α·V_B·PD
        pd = float(
            leland_toft_pd(
                **{k: kw[k] for k in ("asset_value", "asset_vol", "coupon", "rf", "tax_rate")}
            )
        )
        VB = float(
            optimal_default_boundary(
                coupon=5.0,
                rf=0.05,
                sigma_asset=0.30,
                tax_rate=0.30,
            )
        )
        levered = (
            100.0
            + 0.30 * 5.0 / 0.05 * (1.0 - pd)  # tax shield PV
            - 0.30 * VB * pd  # bankruptcy cost PV
        )
        np.testing.assert_allclose(E + D, levered, rtol=1e-9)

    def test_rejects_bad_bankruptcy_cost(self) -> None:
        with pytest.raises(MertonInputError):
            leland_toft_equity_value(
                asset_value=100,
                asset_vol=0.3,
                coupon=5,
                rf=0.05,
                bankruptcy_cost=1.5,
            )


class TestLelandToftModel:
    def test_fit_smoke(self) -> None:
        firm = Firm(
            equity=100.0, debt_short=20, debt_long=30, equity_vol=0.30, rf=0.04, horizon=1.0
        )
        result = LelandToftModel(coupon=2.5, tax_rate=0.25, bankruptcy_cost=0.30).fit(firm)
        assert result.method == "leland_toft"
        assert 0.0 <= result.pd <= 1.0
        assert result.default_point > 0  # V_B*

    def test_requires_equity_vol(self) -> None:
        firm = Firm(equity=100.0, debt_short=20, debt_long=30)
        with pytest.raises(MertonInputError):
            LelandToftModel(coupon=2.5).fit(firm)

    def test_higher_coupon_higher_default_boundary(self) -> None:
        firm = Firm(
            equity=100.0, debt_short=20, debt_long=30, equity_vol=0.30, rf=0.04, horizon=1.0
        )
        low = LelandToftModel(coupon=1.0).fit(firm)
        hi = LelandToftModel(coupon=4.0).fit(firm)
        assert hi.default_point > low.default_point
        assert hi.pd >= low.pd
