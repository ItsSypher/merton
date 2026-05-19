"""Coverage of MertonModel paths not exercised elsewhere."""

from __future__ import annotations

import numpy as np
import pytest

from merton import Firm, MertonModel, fit


class TestPhysicalMeasurePath:
    def test_physical_measure_with_sharpe(self, simple_firm: Firm) -> None:
        result = MertonModel(
            method="jmr_iterative",
            physical_measure=True,
            sharpe_ratio=0.5,
        ).fit(simple_firm)
        assert result.method.endswith("+physical")
        assert 0.0 <= float(result.pd) <= 1.0
        # Risk-neutral PD should be stored on diagnostics for transparency.
        assert "rn_pd" in result.diagnostics_

    def test_physical_measure_no_sharpe_skips(self, simple_firm: Firm) -> None:
        # If physical_measure=True but sharpe is None, the result remains
        # risk-neutral (no error).
        result = MertonModel(
            method="jmr_iterative",
            physical_measure=True,
            sharpe_ratio=None,
        ).fit(simple_firm)
        assert "+physical" not in result.method


class TestGetSetParams:
    def test_get_params_returns_full_config(self, simple_firm: Firm) -> None:
        m = MertonModel(method="naive", tol=1e-5, max_iter=42)
        params = m.get_params()
        assert params["method"] == "naive"
        assert params["tol"] == 1e-5
        assert params["max_iter"] == 42

    def test_set_params_known_keys(self) -> None:
        m = MertonModel(method="jmr_iterative")
        m.set_params(max_iter=999, tol=1e-12)
        assert m.max_iter == 999
        assert m.tol == 1e-12

    def test_set_params_unknown_key_raises(self) -> None:
        m = MertonModel(method="naive")
        with pytest.raises(ValueError):
            m.set_params(does_not_exist=True)


class TestBootstrapShortSeries:
    def test_short_series_returns_none(self) -> None:
        # A snapshot Firm (scalar equity) shouldn't trigger the bootstrap.
        firm = Firm(equity=100.0, debt_short=20.0, debt_long=30.0, equity_vol=0.30)
        result = MertonModel(method="jmr_iterative", n_bootstrap=50).fit(firm)
        assert result.diagnostics_.get("bootstrap") is None


class TestFunctionalShortcut:
    def test_fit_dispatches_to_model(self, simple_firm: Firm) -> None:
        a = fit(simple_firm, method="jmr_iterative")
        b = MertonModel(method="jmr_iterative").fit(simple_firm)
        np.testing.assert_allclose(a.asset_value, b.asset_value, rtol=1e-12)
