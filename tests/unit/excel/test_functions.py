"""Pure-Python tests of the merton.excel.functions wrappers.

These exercise every formula's math without booting a server or Excel.
"""

from __future__ import annotations

import numpy as np
import pytest

from merton.excel.functions import (
    EXCEL_FUNCTIONS,
    merton_asset_value,
    merton_asset_vol,
    merton_backtest,
    merton_black_cox,
    merton_dd,
    merton_greeks,
    merton_pd,
    merton_pd_term,
    merton_portfolio_var,
    merton_spread,
)


class TestSingleFirmFormulas:
    def test_dd_matches_underlying(self) -> None:
        from merton import distance_to_default
        from merton.calibration import jmr_iterative

        res = jmr_iterative(equity=100, equity_vol=0.30, debt=35, rf=0.04, T=1.0)
        dd_ref = float(distance_to_default(res.asset_value, res.asset_vol, 35, 0.04, 1.0))
        assert merton_dd(100, 0.30, 35, 0.04, 1.0) == pytest.approx(dd_ref, rel=1e-10)

    def test_pd_in_unit_interval(self) -> None:
        pd = merton_pd(100, 0.30, 35, 0.04, 1.0)
        assert 0.0 <= pd <= 1.0

    def test_pd_consistent_with_dd(self) -> None:
        from scipy.stats import norm

        dd = merton_dd(100, 0.30, 35, 0.04, 1.0)
        pd = merton_pd(100, 0.30, 35, 0.04, 1.0)
        assert pd == pytest.approx(float(norm.cdf(-dd)), rel=1e-10)

    def test_spread_positive(self) -> None:
        sp = merton_spread(50, 0.45, 80, 0.04, 1.0, 0.6)
        assert sp > 0

    def test_asset_value_and_vol_invert_to_input_equity(self) -> None:
        from merton import equity_value

        a = merton_asset_value(100, 0.30, 35, 0.04, 1.0)
        s = merton_asset_vol(100, 0.30, 35, 0.04, 1.0)
        # Round-trip: plug (a, s) back through BSM and recover the equity.
        e_back = float(equity_value(a, s, 35, 0.04, 1.0))
        assert e_back == pytest.approx(100.0, rel=1e-6)


class TestGreeks:
    def test_greeks_shape(self) -> None:
        g = merton_greeks(100, 0.30, 35, 0.04, 1.0)
        assert isinstance(g, list)
        assert len(g) == 2  # header + values row
        assert len(g[0]) == 6
        assert len(g[1]) == 6
        # Labels in the expected order.
        assert g[0][0] == "delta"
        assert g[0][2] == "vega"
        assert g[0][4] == "rho"

    def test_greeks_finite(self) -> None:
        g = merton_greeks(50, 0.45, 80, 0.04, 1.0)  # leveraged firm
        for v in g[1]:
            assert np.isfinite(v)


class TestPDTerm:
    def test_shape(self) -> None:
        ts = merton_pd_term(100, 0.30, 35, 0.04, [0.25, 0.5, 1.0, 3.0, 5.0])
        assert ts[0] == ["horizon_years", "pd"]
        assert len(ts) == 6
        for row in ts[1:]:
            assert len(row) == 2
            assert 0.0 <= row[1] <= 1.0

    def test_monotone_in_horizon(self) -> None:
        ts = merton_pd_term(100, 0.30, 35, 0.04, [0.25, 0.5, 1.0, 3.0, 5.0])
        pds = [row[1] for row in ts[1:]]
        for i in range(len(pds) - 1):
            assert pds[i] <= pds[i + 1] + 1e-12


class TestBacktest:
    def test_auc_is_max_for_perfect(self) -> None:
        preds = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        defs = np.array([0, 0, 0, 1, 1, 1])
        assert merton_backtest(preds, defs, "AUC") == pytest.approx(1.0)

    def test_brier_zero_for_perfect(self) -> None:
        preds = np.array([0.0, 0.0, 1.0, 1.0])
        defs = np.array([0, 0, 1, 1])
        assert merton_backtest(preds, defs, "Brier") == pytest.approx(0.0)

    def test_ks_metric(self) -> None:
        preds = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        defs = np.array([0, 0, 0, 1, 1, 1])
        assert merton_backtest(preds, defs, "KS") == pytest.approx(1.0)

    def test_accuracy_ratio_alias(self) -> None:
        preds = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        defs = np.array([0, 0, 0, 1, 1, 1])
        ar = merton_backtest(preds, defs, "AccuracyRatio")
        gini = merton_backtest(preds, defs, "Gini")
        ar2 = merton_backtest(preds, defs, "AR")
        assert ar == pytest.approx(gini) == pytest.approx(ar2)

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown metric"):
            merton_backtest(np.array([0.1]), np.array([0]), "BOGUS")


class TestPortfolioAndBlackCox:
    def test_portfolio_var_basel_path(self) -> None:
        v = merton_portfolio_var(0.02, 0.45)
        assert 0 < v < 1

    def test_portfolio_var_explicit_rho(self) -> None:
        v = merton_portfolio_var(0.02, 0.45, 0.15)
        assert 0 < v < 1

    def test_black_cox_dominates_merton_pd(self) -> None:
        m_pd = merton_pd(50, 0.45, 80, 0.04, 1.0)
        bc_pd = merton_black_cox(50, 0.45, 80, 0.04, 1.0)
        assert bc_pd >= m_pd - 1e-10

    def test_black_cox_growth_rate_increases_pd(self) -> None:
        flat = merton_black_cox(80, 0.40, 80, 0.04, 1.0, 0.0)
        grow = merton_black_cox(80, 0.40, 80, 0.04, 1.0, 0.05)
        assert grow >= flat


class TestRegistry:
    def test_all_callables_in_registry(self) -> None:
        names = {row[0] for row in EXCEL_FUNCTIONS}
        # Every expected formula appears exactly once.
        expected = {
            "MERTON_DD",
            "MERTON_PD",
            "MERTON_SPREAD",
            "MERTON_ASSET_VALUE",
            "MERTON_ASSET_VOL",
            "MERTON_GREEKS",
            "MERTON_PD_TERM",
            "MERTON_BACKTEST",
            "MERTON_PORTFOLIO_VAR",
            "MERTON_BLACK_COX",
        }
        assert names == expected
