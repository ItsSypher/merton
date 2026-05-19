"""JAX autodiff Greeks must agree with the closed-form implementations."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from merton.greeks import autodiff as ad
from merton.greeks import (
    equity_delta,
    equity_gamma,
    equity_rho,
    equity_theta,
    equity_vega,
    pd_leverage_sensitivity,
    pd_rate_sensitivity,
    pd_vol_sensitivity,
)

A = np.array([80.0, 100.0, 150.0, 200.0])
S = np.array([0.20, 0.25, 0.30, 0.35])
D = np.array([50.0, 60.0, 70.0, 80.0])
R = 0.04
T = 1.0


@pytest.mark.jax
@pytest.mark.parametrize(
    "closed_form,autodiff_fn",
    [
        (equity_delta, ad.equity_delta_ad),
        (equity_vega, ad.equity_vega_ad),
        (equity_gamma, ad.equity_gamma_ad),
        (equity_rho, ad.equity_rho_ad),
        (equity_theta, ad.equity_theta_ad),
        (pd_leverage_sensitivity, ad.pd_leverage_sensitivity_ad),
        (pd_vol_sensitivity, ad.pd_vol_sensitivity_ad),
        (pd_rate_sensitivity, ad.pd_rate_sensitivity_ad),
    ],
)
def test_autodiff_matches_closed_form(closed_form, autodiff_fn) -> None:
    cf = np.asarray(closed_form(A, S, D, R, T))
    ad_v = np.asarray(autodiff_fn(A, S, D, R, T))
    np.testing.assert_allclose(ad_v, cf, rtol=1e-5, atol=1e-9)
