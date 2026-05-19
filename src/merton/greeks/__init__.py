"""First- and second-order sensitivities for the Merton model.

All Greeks here are closed-form expressions derived from Black-Scholes-Merton.
Numerical / autodiff Greeks (via JAX) live in :mod:`merton.greeks.autodiff`
behind the ``[jax]`` extra.
"""

from __future__ import annotations

from .equity import (
    GreeksResult,
    equity_delta,
    equity_gamma,
    equity_rho,
    equity_theta,
    equity_vega,
    greeks,
)
from .pd_sensitivity import (
    pd_leverage_sensitivity,
    pd_rate_sensitivity,
    pd_vol_sensitivity,
)
from .spread_sensitivity import spread_sensitivity

__all__ = [
    "GreeksResult",
    "equity_delta",
    "equity_gamma",
    "equity_rho",
    "equity_theta",
    "equity_vega",
    "greeks",
    "pd_leverage_sensitivity",
    "pd_rate_sensitivity",
    "pd_vol_sensitivity",
    "spread_sensitivity",
]
