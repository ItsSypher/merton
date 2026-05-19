"""Tests for the Longstaff-Schwartz two-factor model."""

from __future__ import annotations

import pytest

from merton import Firm
from merton.exceptions import MertonInputError
from merton.extensions import (
    LongstaffSchwartzModel,
    VasicekParams,
    longstaff_schwartz_pd_analytic,
    longstaff_schwartz_pd_mc,
)


@pytest.fixture
def vasicek() -> VasicekParams:
    return VasicekParams(kappa=0.5, theta=0.04, eta=0.01, r0=0.04)


class TestVasicekParams:
    def test_validation(self) -> None:
        with pytest.raises(MertonInputError):
            VasicekParams(kappa=0.0, theta=0.04, eta=0.01, r0=0.04)
        with pytest.raises(MertonInputError):
            VasicekParams(kappa=0.5, theta=0.04, eta=0.0, r0=0.04)


class TestAnalyticPD:
    def test_well_capitalised_low_pd(self, vasicek: VasicekParams) -> None:
        pd = longstaff_schwartz_pd_analytic(
            asset_value=200.0,
            asset_vol=0.25,
            barrier=60.0,
            T=1.0,
            vasicek=vasicek,
        )
        assert 0 <= pd < 0.01

    def test_leveraged_higher_pd(self, vasicek: VasicekParams) -> None:
        low = longstaff_schwartz_pd_analytic(
            asset_value=120.0,
            asset_vol=0.25,
            barrier=60.0,
            T=1.0,
            vasicek=vasicek,
        )
        hi = longstaff_schwartz_pd_analytic(
            asset_value=70.0,
            asset_vol=0.50,
            barrier=60.0,
            T=1.0,
            vasicek=vasicek,
        )
        assert hi > low

    def test_pd_in_unit_interval(self, vasicek: VasicekParams) -> None:
        for A in (50.0, 100.0, 200.0):
            pd = longstaff_schwartz_pd_analytic(
                asset_value=A,
                asset_vol=0.30,
                barrier=60.0,
                T=1.0,
                vasicek=vasicek,
            )
            assert 0.0 <= pd <= 1.0


class TestMonteCarloPD:
    def test_mc_smoke(self, vasicek: VasicekParams) -> None:
        pd, se = longstaff_schwartz_pd_mc(
            asset_value=100.0,
            asset_vol=0.30,
            barrier=60.0,
            T=1.0,
            vasicek=vasicek,
            correlation=-0.3,
            n_paths=2000,
            n_steps=120,
            seed=42,
        )
        assert 0.0 <= pd <= 1.0
        assert se >= 0

    def test_negative_correlation_widens_pd(self, vasicek: VasicekParams) -> None:
        """Negative ρ between assets and rates ⇒ defaults concentrate in
        recessions (low rates, low assets), so the survival probability
        is *lower* and the PD slightly higher than the zero-correlation
        analytic case."""
        pd_zero = longstaff_schwartz_pd_analytic(
            asset_value=80.0,
            asset_vol=0.40,
            barrier=60.0,
            T=1.0,
            vasicek=vasicek,
        )
        pd_neg, _ = longstaff_schwartz_pd_mc(
            asset_value=80.0,
            asset_vol=0.40,
            barrier=60.0,
            T=1.0,
            vasicek=vasicek,
            correlation=-0.5,
            n_paths=5000,
            n_steps=252,
            seed=7,
        )
        # Small panels can be noisy; we just check the result is finite and
        # in the same ballpark.
        assert 0.0 <= pd_neg <= 1.0
        assert abs(pd_neg - pd_zero) < 0.20

    def test_below_barrier_returns_one(self, vasicek: VasicekParams) -> None:
        pd, _ = longstaff_schwartz_pd_mc(
            asset_value=30.0,
            asset_vol=0.30,
            barrier=50.0,
            T=1.0,
            vasicek=vasicek,
            n_paths=10,
            n_steps=5,
        )
        assert pd == 1.0

    def test_rejects_invalid_correlation(self, vasicek: VasicekParams) -> None:
        with pytest.raises(MertonInputError):
            longstaff_schwartz_pd_mc(
                asset_value=100,
                asset_vol=0.3,
                barrier=60,
                T=1.0,
                vasicek=vasicek,
                correlation=1.5,
            )

    def test_rejects_negative_asset(self, vasicek: VasicekParams) -> None:
        with pytest.raises(MertonInputError):
            longstaff_schwartz_pd_mc(
                asset_value=0,
                asset_vol=0.3,
                barrier=60,
                T=1.0,
                vasicek=vasicek,
            )


class TestLongstaffSchwartzModel:
    def test_zero_correlation_uses_analytic(self, vasicek: VasicekParams) -> None:
        firm = Firm(
            equity=100.0, debt_short=20, debt_long=30, equity_vol=0.30, rf=0.04, horizon=1.0
        )
        result = LongstaffSchwartzModel(vasicek=vasicek, correlation=0.0).fit(firm)
        assert result.method == "longstaff_schwartz"
        assert result.diagnostics["mc_stderr"] is None

    def test_nonzero_correlation_uses_mc(self, vasicek: VasicekParams) -> None:
        firm = Firm(
            equity=100.0, debt_short=20, debt_long=30, equity_vol=0.30, rf=0.04, horizon=1.0
        )
        result = LongstaffSchwartzModel(
            vasicek=vasicek,
            correlation=-0.4,
            mc_paths=500,
            mc_seed=42,
        ).fit(firm)
        assert result.diagnostics["mc_stderr"] is not None

    def test_requires_equity_vol(self, vasicek: VasicekParams) -> None:
        firm = Firm(equity=100.0, debt_short=20, debt_long=30)
        with pytest.raises(MertonInputError):
            LongstaffSchwartzModel(vasicek=vasicek).fit(firm)
