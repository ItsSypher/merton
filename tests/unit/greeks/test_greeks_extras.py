"""Extra coverage: GreeksResult exporters, dividend-yield path."""

from __future__ import annotations

import numpy as np

from merton.greeks import greeks


class TestGreeksResultExports:
    def test_to_dict(self) -> None:
        g = greeks(100.0, 0.25, 60.0, 0.04, 1.0)
        d = g.to_dict()
        for k in (
            "equity_delta",
            "equity_gamma",
            "equity_vega",
            "equity_theta",
            "equity_rho",
            "pd_dleverage",
            "pd_dvol",
            "pd_drate",
        ):
            assert k in d

    def test_to_pandas(self) -> None:
        g = greeks(100.0, 0.25, 60.0, 0.04, 1.0)
        df = g.to_pandas()
        assert len(df) == 1
        assert "equity_delta" in df.columns

    def test_with_dividend_yield(self) -> None:
        # The q > 0 branches in equity_delta / equity_theta /
        # equity_gamma / equity_vega should remain finite.
        g_q0 = greeks(100.0, 0.25, 60.0, 0.04, 1.0, dividend_yield=0.0)
        g_q3 = greeks(100.0, 0.25, 60.0, 0.04, 1.0, dividend_yield=0.03)
        # Dividends shift the equity-as-call profile; delta must drop.
        assert float(g_q3.equity_delta) < float(g_q0.equity_delta)
        # Theta must remain finite.
        assert np.isfinite(float(g_q3.equity_theta))
