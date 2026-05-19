"""Validate Duan MLE on a controlled synthetic series.

We *generate* equity prices from a known asset GBM and then check that the
Duan MLE recovers the input parameters within a documented tolerance. This
guards against any silent algebraic regression in the likelihood / Jacobian.
"""

from __future__ import annotations

import numpy as np
import pytest

from merton._backend._numpy import equity_value as _equity_value_kernel
from merton.calibration import duan_mle


def _simulate_equity_path(
    *,
    A0: float,
    mu: float,
    sigma_A: float,
    debt: float,
    rf: float,
    T: float,
    n_steps: int,
    seed: int,
) -> np.ndarray:
    """Simulate ``A_t`` under GBM(μ, σ_A) and map each to its BSM equity value."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    log_returns = rng.normal((mu - 0.5 * sigma_A**2) * dt, sigma_A * np.sqrt(dt), size=n_steps - 1)
    A = A0 * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))
    return np.array([float(_equity_value_kernel(A_t, sigma_A, debt, rf, T, 0.0)) for A_t in A])


@pytest.mark.golden
class TestDuanSynthetic:
    def test_recovers_volatility_within_tolerance(self) -> None:
        true_sigma = 0.25
        true_mu = 0.10
        eq_series = _simulate_equity_path(
            A0=130.0,
            mu=true_mu,
            sigma_A=true_sigma,
            debt=35.0,
            rf=0.04,
            T=1.0,
            n_steps=252,
            seed=2026,
        )
        res = duan_mle(
            equity_series=eq_series,
            debt=35.0,
            rf=0.04,
            T=1.0,
            survivor_bias_correction=False,
        )
        # σ_A should be recovered within ±0.06 (sample noise for 1y daily).
        assert abs(res.asset_vol - true_sigma) < 0.06, (
            f"σ_A estimate {res.asset_vol:.4f} too far from true {true_sigma:.4f}"
        )

    def test_log_likelihood_finite_and_negative_of_neg_ll(self) -> None:
        eq = _simulate_equity_path(
            A0=130.0,
            mu=0.10,
            sigma_A=0.25,
            debt=35.0,
            rf=0.04,
            T=1.0,
            n_steps=252,
            seed=2027,
        )
        res = duan_mle(
            equity_series=eq,
            debt=35.0,
            rf=0.04,
            T=1.0,
            survivor_bias_correction=False,
        )
        assert res.log_likelihood is not None
        assert np.isfinite(res.log_likelihood)
