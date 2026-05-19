"""Tests for the Bayesian MCMC calibrator."""

from __future__ import annotations

import numpy as np
import pytest

emcee = pytest.importorskip("emcee")

from merton import Firm, equity_value  # noqa: E402
from merton.calibration import bayesian_mcmc  # noqa: E402
from merton.calibration.bayesian_mcmc import BayesianMCMCCalibrator  # noqa: E402
from merton.exceptions import InsufficientDataError, MertonInputError  # noqa: E402


def _synthetic_equity_series(seed: int = 42, n: int = 252) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu_true, sigma_true = 0.10, 0.25
    dt = 1 / 252
    log_ret = rng.normal((mu_true - 0.5 * sigma_true**2) * dt, sigma_true * np.sqrt(dt), size=n)
    A_path = 130.0 * np.exp(np.cumsum(log_ret))
    return np.array([float(equity_value(A, sigma_true, 35.0, 0.04, 1.0)) for A in A_path])


class TestBayesianMCMC:
    @pytest.mark.slow
    def test_recovers_volatility(self) -> None:
        eq = _synthetic_equity_series()
        res = bayesian_mcmc(
            equity_series=eq,
            debt=35.0,
            rf=0.04,
            T=1.0,
            n_walkers=24,
            n_steps=400,
            burn_in=100,
            thin=2,
            seed=7,
        )
        # MAP within ±0.05 of true σ_A = 0.25.
        assert abs(res.asset_vol - 0.25) < 0.05
        assert res.chain.shape[1] == 2
        assert 0.0 < res.acceptance_fraction < 1.0

    @pytest.mark.slow
    def test_credible_intervals_contain_truth(self) -> None:
        eq = _synthetic_equity_series(seed=7)
        res = bayesian_mcmc(
            equity_series=eq,
            debt=35.0,
            rf=0.04,
            T=1.0,
            n_walkers=24,
            n_steps=400,
            burn_in=100,
            thin=2,
            seed=1,
        )
        ci = res.credible_interval("asset_vol", level=0.95)
        assert ci.lower < 0.25 < ci.upper

    @pytest.mark.slow
    def test_rejects_unknown_param(self) -> None:
        eq = _synthetic_equity_series()
        res = bayesian_mcmc(
            equity_series=eq,
            debt=35.0,
            rf=0.04,
            T=1.0,
            n_walkers=24,
            n_steps=200,
            burn_in=50,
            thin=2,
            seed=0,
        )
        with pytest.raises(MertonInputError):
            res.credible_interval("bogus")

    def test_rejects_short_series(self) -> None:
        with pytest.raises(InsufficientDataError):
            bayesian_mcmc(
                equity_series=np.array([100.0, 101.0]),
                debt=35.0,
                rf=0.04,
                T=1.0,
            )

    def test_rejects_bad_walkers_or_steps(self) -> None:
        eq = _synthetic_equity_series()
        with pytest.raises(MertonInputError):
            bayesian_mcmc(equity_series=eq, debt=35.0, rf=0.04, T=1.0, n_walkers=2)
        with pytest.raises(MertonInputError):
            bayesian_mcmc(
                equity_series=eq,
                debt=35.0,
                rf=0.04,
                T=1.0,
                n_steps=50,
                burn_in=100,
            )


class TestBayesianMCMCCalibrator:
    @pytest.mark.slow
    def test_via_calibrator_class(self) -> None:
        eq = _synthetic_equity_series()
        firm = Firm(equity=eq, debt_short=20, debt_long=30, rf=0.04, horizon=1.0)
        res = BayesianMCMCCalibrator(n_steps=300, burn_in=80, seed=11).fit(firm)
        assert res.method == "bayesian_mcmc"
        assert res.chain.shape[1] == 2

    def test_rejects_scalar_firm(self) -> None:
        firm = Firm(equity=100.0, debt_short=20, debt_long=30)
        with pytest.raises(InsufficientDataError):
            BayesianMCMCCalibrator().fit(firm)
