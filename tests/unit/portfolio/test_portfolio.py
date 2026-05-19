"""Tests for the Portfolio container and Monte Carlo / analytic engines."""

from __future__ import annotations

import numpy as np
import pytest

from merton import Firm
from merton.exceptions import MertonInputError
from merton.portfolio import LossDistribution, Portfolio


def _make_firms(n: int = 10) -> list[Firm]:
    rng = np.random.default_rng(0)
    return [
        Firm(
            equity=float(rng.uniform(80, 200)),
            debt_short=float(rng.uniform(10, 40)),
            debt_long=float(rng.uniform(20, 60)),
            equity_vol=float(rng.uniform(0.20, 0.45)),
            rf=0.04,
            ticker=f"F{i}",
        )
        for i in range(n)
    ]


class TestPortfolioConstruction:
    def test_from_firm_list(self) -> None:
        firms = _make_firms(5)
        pf = Portfolio(firms, exposures=np.ones(5), lgd=0.5)
        assert pf.firms is not None
        assert pf.pds.shape == (5,)
        np.testing.assert_array_equal(pf.lgds, np.full(5, 0.5))

    def test_from_pds_array(self) -> None:
        pf = Portfolio(np.array([0.01, 0.02, 0.05]))
        assert pf.firms is None
        np.testing.assert_array_equal(pf.pds, [0.01, 0.02, 0.05])

    def test_fit_populates_pds(self) -> None:
        firms = _make_firms(5)
        pf = Portfolio(firms)
        res = pf.fit(method="jmr_iterative")
        assert not np.any(np.isnan(pf.pds))
        assert res.expected_loss >= 0

    def test_pds_only_fit_returns_uncalibrated_result(self) -> None:
        pf = Portfolio(np.array([0.02, 0.03]), lgd=0.5)
        res = pf.fit()
        assert res.diagnostics["calibrated"] is False
        assert res.expected_loss == pytest.approx((0.02 + 0.03) * 0.5)

    def test_simulate_without_fit_raises(self) -> None:
        firms = _make_firms(3)
        pf = Portfolio(firms)
        with pytest.raises(MertonInputError):
            pf.simulate(n_sims=100)


class TestPortfolioSimulation:
    def test_simulate_pds_only(self) -> None:
        pf = Portfolio(np.full(50, 0.05), exposures=np.ones(50), lgd=0.5)
        pf.fit()
        dist = pf.simulate(n_sims=5_000, seed=42)
        assert isinstance(dist, LossDistribution)
        assert len(dist.losses) == 5_000  # antithetic round-trip

    def test_record_contributions(self) -> None:
        pf = Portfolio(np.full(20, 0.10), exposures=np.ones(20), lgd=1.0)
        pf.fit()
        dist = pf.simulate(n_sims=2_000, seed=1, record_contributions=True)
        assert dist.contributions is not None
        assert dist.contributions.shape == (2_000, 20)
        contribs = dist.firm_contributions(level=0.95, method="es")
        assert contribs.shape == (20,)
        # Each contribution must be ≤ LGD*exposure.
        assert np.all(contribs <= 1.0)

    def test_t_copula_path(self) -> None:
        pf = Portfolio(np.full(10, 0.1), correlation=0.3, copula="t", df=5, lgd=1.0)
        pf.fit()
        dist = pf.simulate(n_sims=1_000, seed=3)
        assert dist.losses.size == 1_000

    def test_unknown_copula(self) -> None:
        pf = Portfolio(np.full(5, 0.1), copula="bogus")
        pf.fit()
        with pytest.raises(MertonInputError):
            pf.simulate(n_sims=100)


class TestPortfolioAnalytic:
    def test_analytic_vasicek(self) -> None:
        pf = Portfolio(np.full(100, 0.02), exposures=np.full(100, 1e6), lgd=0.45)
        pf.fit()
        ana = pf.analytic_vasicek(confidence=0.999)
        assert ana["VaR"] > 0
        assert ana["unexpected_loss"] >= 0
        assert ana["expected_loss"] > 0

    def test_analytic_requires_pds(self) -> None:
        firms = _make_firms(3)
        pf = Portfolio(firms)
        with pytest.raises(MertonInputError):
            pf.analytic_vasicek()  # pds still NaN before fit


class TestImmutableAdd:
    def test_add_firm_returns_new_portfolio(self) -> None:
        firms = _make_firms(3)
        pf = Portfolio(firms, exposures=np.ones(3), lgd=0.5)
        pf2 = pf.add_firm(_make_firms(1)[0], exposure=2.0, lgd=0.6)
        assert pf2 is not pf
        assert len(pf2.pds) == 4

    def test_add_firm_on_pds_only_raises(self) -> None:
        pf = Portfolio(np.array([0.02]))
        with pytest.raises(MertonInputError):
            pf.add_firm(_make_firms(1)[0])
