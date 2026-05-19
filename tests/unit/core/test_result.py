"""Tests for the MertonResult container."""

from __future__ import annotations

import pandas as pd

from merton import Firm, fit


class TestMertonResult:
    def test_to_dict(self, simple_firm: Firm) -> None:
        result = fit(simple_firm, method="jmr_iterative")
        d = result.to_dict()
        for k in ("dd", "pd", "asset_value", "asset_vol", "method"):
            assert k in d

    def test_to_pandas(self, simple_firm: Firm) -> None:
        result = fit(simple_firm, method="jmr_iterative")
        df = result.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_implied_spread(self, simple_firm: Firm) -> None:
        result = fit(simple_firm, method="jmr_iterative")
        s = result.implied_spread(lgd=0.6)
        assert float(s) >= 0

    def test_greeks(self, simple_firm: Firm) -> None:
        result = fit(simple_firm, method="jmr_iterative")
        g = result.greeks()
        assert 0 < float(g.equity_delta) < 1
        assert float(g.equity_vega) > 0
        assert float(g.equity_gamma) > 0

    def test_physical_pd_lowers_for_positive_sharpe(self, simple_firm: Firm) -> None:
        result = fit(simple_firm, method="jmr_iterative")
        rn = float(result.pd)
        physical = float(result.physical_pd(sharpe_ratio=0.5))
        # Positive Sharpe ratio implies firm assets drift up faster than r,
        # so physical PD < risk-neutral PD.
        assert physical < rn

    def test_repr_html(self, simple_firm: Firm) -> None:
        result = fit(simple_firm, method="jmr_iterative")
        html = result._repr_html_()
        assert "MertonResult" in html
        assert "<table>" in html

    def test_summary_contains_key_metrics(self, simple_firm: Firm) -> None:
        result = fit(simple_firm, method="jmr_iterative")
        text = result.summary()
        for line in (
            "Distance-to-default",
            "Probability of default",
            "Asset value",
            "Asset volatility",
        ):
            assert line in text

    def test_pd_term_structure_shape(self, simple_firm: Firm) -> None:
        result = fit(simple_firm, method="jmr_iterative")
        ts = result.pd_term_structure(horizons=[0.5, 1.0, 2.0])
        assert ts.shape == (3, 3)
