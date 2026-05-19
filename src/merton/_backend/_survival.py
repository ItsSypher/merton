"""First-passage / survival probabilities for geometric Brownian motion.

The Duan-Gauthier-Simonato-Zaanoun (2004) survivorship-bias correction needs
the probability that asset value ``A_t`` *never* falls below the default
threshold ``D`` over an observation window ``[0, T]``, given the calibrated
parameters ``μ`` and ``σ_A``. The closed form is the standard
reflection-principle result for Brownian motion with drift.

Notation
--------
Let ``X_t = log(A_t)``. Then ``X`` is Brownian motion with drift
``ν = μ - σ²/2`` and volatility ``σ``. Define ``a = log(A_0 / D) > 0``.

The probability that ``X`` stays above ``log D`` for all ``t in [0, T]`` is

.. math::

    P(\\min_{0 \\le t \\le T} X_t > \\log D \\mid A_0, \\mu, \\sigma_A)
    = \\Phi\\!\\left(\\frac{a + \\nu T}{\\sigma\\sqrt{T}}\\right)
      - e^{-2\\nu a / \\sigma^2}\\,
        \\Phi\\!\\left(\\frac{-a + \\nu T}{\\sigma\\sqrt{T}}\\right).

The function returns the log of this probability for numerical stability;
the survivorship correction adds ``-log P(survive)`` to the negative
log-likelihood the optimiser is minimising.
"""

from __future__ import annotations

import numpy as np

from ._numpy import norm_cdf

_EPS = 1e-300


def log_survival_probability(
    asset_value_0: float,
    debt: float,
    drift: float,
    sigma: float,
    T: float,
) -> float:
    """Log P(min_{0≤t≤T} A_t > D | A_0, μ, σ).

    Closed-form reflection-principle expression for geometric Brownian motion.
    Returns ``log P`` (always ≤ 0). When ``A_0 ≤ D`` the survival probability
    is zero; we return a large negative finite number instead of ``-inf`` so
    optimisers see a smooth, descent-friendly objective near the boundary.
    """
    a = float(np.log(asset_value_0) - np.log(debt))
    if a <= 0:
        return -1e6
    nu = drift - 0.5 * sigma * sigma
    sqrtT = float(np.sqrt(T))
    z1 = (a + nu * T) / (sigma * sqrtT)
    z2 = (-a + nu * T) / (sigma * sqrtT)
    p = float(norm_cdf(z1) - np.exp(-2.0 * nu * a / (sigma * sigma)) * norm_cdf(z2))
    # Clip negative numerical artefacts (can happen for tiny p).
    p = max(p, _EPS)
    return float(np.log(p))


def survival_probability(
    asset_value_0: float,
    debt: float,
    drift: float,
    sigma: float,
    T: float,
) -> float:
    """P(min_{0≤t≤T} A_t > D | A_0, μ, σ)."""
    return float(np.exp(log_survival_probability(asset_value_0, debt, drift, sigma, T)))


__all__ = ["log_survival_probability", "survival_probability"]
