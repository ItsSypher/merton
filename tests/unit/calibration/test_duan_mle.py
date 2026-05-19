"""Tests for the Duan MLE calibrator."""

from __future__ import annotations

import numpy as np
import pytest

from merton import Firm, MertonModel, fit
from merton.calibration import duan_mle
from merton.exceptions import InsufficientDataError


@pytest.fixture
def gbm_equity_series() -> np.ndarray:
    """A synthetic 1-year daily equity series under known parameters."""
    rng = np.random.default_rng(7)
    n = 252
    sigma_E = 0.30
    mu_E = 0.08
    dt = 1.0 / 252.0
    log_ret = rng.normal((mu_E - 0.5 * sigma_E**2) * dt, sigma_E * np.sqrt(dt), size=n)
    return 100.0 * np.exp(np.cumsum(log_ret))


class TestDuanMLE:
    def test_runs_on_short_synthetic_series(self, gbm_equity_series) -> None:
        res = duan_mle(
            equity_series=gbm_equity_series,
            debt=35.0,
            rf=0.04,
            T=1.0,
            survivor_bias_correction=False,  # faster, no log-survival probe
        )
        assert res.converged
        assert 0.05 < res.asset_vol < 1.0
        assert res.asset_drift is not None
        assert res.covariance is not None
        assert res.covariance.shape == (2, 2)
        assert res.log_likelihood is not None

    def test_via_model(self, gbm_equity_series) -> None:
        firm = Firm(equity=gbm_equity_series, debt_short=20, debt_long=30, rf=0.04, horizon=1.0)
        result = MertonModel(method="duan_mle").fit(firm)
        assert result.method == "duan_mle"
        assert result.converged
        assert 0 <= float(result.pd) <= 1
        assert result.covariance_ is not None

    def test_rejects_short_series(self) -> None:
        firm = Firm(equity=np.array([100.0, 101.0]), debt_short=20, debt_long=30, rf=0.04)
        with pytest.raises(InsufficientDataError):
            MertonModel(method="duan_mle").fit(firm)

    def test_asymptotic_ci(self, gbm_equity_series) -> None:
        firm = Firm(equity=gbm_equity_series, debt_short=20, debt_long=30, rf=0.04, horizon=1.0)
        result = fit(firm, method="duan_mle")
        ci = result.confidence_interval(level=0.95, method="asymptotic")
        # All expected keys present.
        for k in ("asset_vol", "asset_drift", "dd", "pd"):
            assert k in ci
            assert ci[k].lower <= ci[k].upper

    def test_recovers_input_volatility_approximately(self, gbm_equity_series) -> None:
        """Sanity: asset σ should be in the ballpark of the equity σ (≈ 0.30 here)
        scaled down by the leverage factor E/(E+D). For E=100, D=35 ⇒ ~0.22."""
        res = duan_mle(
            equity_series=gbm_equity_series,
            debt=35.0,
            rf=0.04,
            T=1.0,
            survivor_bias_correction=False,
        )
        assert 0.10 < res.asset_vol < 0.35


class TestSurvivorshipCorrection:
    """The correction shifts the optimum but should not break finiteness."""

    def test_correction_runs(self, gbm_equity_series) -> None:
        res = duan_mle(
            equity_series=gbm_equity_series,
            debt=35.0,
            rf=0.04,
            T=1.0,
            survivor_bias_correction=True,
        )
        assert res.converged
        assert np.isfinite(res.asset_vol)
        assert res.diagnostics["survivor_bias_correction"] is True
