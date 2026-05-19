"""Tests for the three Phase 0.1 calibrators."""

from __future__ import annotations

import numpy as np
import pytest

from merton import Firm, MertonModel, fit
from merton.calibration import (
    available_methods,
    jmr_iterative,
    naive,
    vassalou_xing,
)
from merton.exceptions import MertonInputError


class TestRegistry:
    def test_three_methods_registered(self) -> None:
        methods = set(available_methods())
        assert {"naive", "jmr_iterative", "vassalou_xing"}.issubset(methods)


class TestNaive:
    def test_naive_runs(self, simple_firm: Firm) -> None:
        res = naive(
            equity=simple_firm.equity,
            equity_vol=simple_firm.equity_vol,
            debt=simple_firm.default_point_value(),
            rf=simple_firm.rf,
            T=simple_firm.horizon,
        )
        assert res.method == "naive"
        assert res.asset_vol > 0
        assert float(res.asset_value) > simple_firm.equity

    def test_via_model(self, simple_firm: Firm) -> None:
        result = MertonModel(method="naive").fit(simple_firm)
        assert result.method == "naive"
        assert 0 <= float(result.pd) <= 1


class TestJMRIterative:
    def test_jmr_runs(self, simple_firm: Firm) -> None:
        res = jmr_iterative(
            equity=float(simple_firm.equity),
            equity_vol=float(simple_firm.equity_vol),
            debt=float(simple_firm.default_point_value()),
            rf=float(simple_firm.rf),
            T=float(simple_firm.horizon),
        )
        assert res.converged
        assert res.asset_vol > 0
        # Asset value must exceed equity (because some assets back the debt)
        assert res.asset_value > simple_firm.equity

    def test_recovers_equity_value(self, simple_firm: Firm) -> None:
        """Plugging the inferred (A, σ_A) back through BSM must recover E exactly."""
        from merton import equity_value

        res = jmr_iterative(
            equity=float(simple_firm.equity),
            equity_vol=float(simple_firm.equity_vol),
            debt=float(simple_firm.default_point_value()),
            rf=float(simple_firm.rf),
            T=float(simple_firm.horizon),
        )
        E_back = float(
            equity_value(
                res.asset_value,
                res.asset_vol,
                simple_firm.default_point_value(),
                simple_firm.rf,
                simple_firm.horizon,
            )
        )
        np.testing.assert_allclose(E_back, simple_firm.equity, rtol=1e-6)

    def test_requires_equity_vol(self) -> None:
        firm = Firm(equity=100, debt_short=20, debt_long=30)  # σ_E omitted
        with pytest.raises(MertonInputError):
            MertonModel(method="jmr_iterative").fit(firm)


class TestVassalouXing:
    def test_snapshot_mode_matches_jmr(self, simple_firm: Firm) -> None:
        vx = vassalou_xing(
            equity=float(simple_firm.equity),
            equity_vol=float(simple_firm.equity_vol),
            debt=float(simple_firm.default_point_value()),
            rf=float(simple_firm.rf),
            T=float(simple_firm.horizon),
        )
        jmr = jmr_iterative(
            equity=float(simple_firm.equity),
            equity_vol=float(simple_firm.equity_vol),
            debt=float(simple_firm.default_point_value()),
            rf=float(simple_firm.rf),
            T=float(simple_firm.horizon),
        )
        np.testing.assert_allclose(vx.asset_value, jmr.asset_value, rtol=1e-5)
        np.testing.assert_allclose(vx.asset_vol, jmr.asset_vol, rtol=1e-5)

    def test_series_mode_converges(self, equity_series_firm: Firm) -> None:
        res = vassalou_xing(
            equity=equity_series_firm.equity,
            debt=float(equity_series_firm.default_point_value()),
            rf=float(equity_series_firm.rf),
            T=float(equity_series_firm.horizon),
        )
        assert res.converged
        assert res.asset_drift is not None
        assert res.diagnostics["mode"] == "series"
        # Asset volatility should be in a plausible range for our GBM-generated series.
        assert 0.05 < res.asset_vol < 1.0


class TestFitOrchestrator:
    def test_functional_fit_default(self, simple_firm: Firm) -> None:
        result = fit(simple_firm)
        assert result.method == "vassalou_xing"
        assert result.converged
        assert 0 <= float(result.pd) <= 1
        assert float(result.dd) > 0

    def test_summary_renders(self, simple_firm: Firm) -> None:
        result = fit(simple_firm, method="jmr_iterative")
        text = result.summary()
        assert "Distance-to-default" in text
        assert "MertonResult" in text

    def test_term_structure(self, simple_firm: Firm) -> None:
        result = fit(simple_firm, method="jmr_iterative")
        ts = result.pd_term_structure(horizons=[0.25, 0.5, 1.0, 3.0])
        assert list(ts.columns) == ["horizon_years", "dd", "pd"]
        assert len(ts) == 4
        # PD monotone-increasing in horizon (for a well-capitalised firm)
        assert all(ts["pd"].diff().dropna() > 0)
