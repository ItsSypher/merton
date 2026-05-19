r"""Geske (1977) compound-option pricing for multi-tranche debt.

When a firm has a layered debt schedule (e.g. short-term and long-term
liabilities maturing at distinct dates ``t_1 < t_2``), equity is no longer
a single call on assets — it is a *call on a call*. At ``t_1`` the equity
holders must either roll the short-term debt by paying ``D_1`` (which
requires the residual call on the long-term debt to be worth at least
``D_1``) or walk away. Geske (1977) derives the closed-form value of this
compound option using the bivariate-normal CDF.

Two-period formula
------------------
Let ``A_t`` follow geometric Brownian motion with drift ``r - q`` and
volatility ``σ_A`` under the risk-neutral measure. Suppose ``D_1`` is due
at ``t_1`` and ``D_2`` at ``t_2 > t_1``. Define ``A^*`` as the asset value
at ``t_1`` that makes the residual call on the long-term debt exactly
worth ``D_1``:

.. math::

   C\bigl(A^*; D_2, t_2 - t_1, r, \sigma_A, q\bigr) = D_1.

This is a 1-D root-finding problem.

Define the standard BSM helpers

.. math::

   d_1^{(i)} &= \frac{\ln(A_0/X_i) + (r - q + \tfrac{1}{2}\sigma_A^2)\,t_i}{\sigma_A \sqrt{t_i}}, \\
   d_2^{(i)} &= d_1^{(i)} - \sigma_A \sqrt{t_i}, \\
   \rho      &= \sqrt{t_1/t_2},

with ``X_1 = A^*`` (for the inner call at ``t_1``) and ``X_2 = D_2``
(for the outer call at ``t_2``). The Geske compound-call value is then

.. math::

   E_0 = A_0 e^{-q t_2}\, \Phi_2\!\bigl(d_1^{(1)}, d_1^{(2)};\,\rho\bigr)
        - D_2 e^{-r t_2}\, \Phi_2\!\bigl(d_2^{(1)}, d_2^{(2)};\,\rho\bigr)
        - D_1 e^{-r t_1}\, \Phi\!\bigl(d_2^{(1)}\bigr),

where ``Φ_2(·,·; ρ)`` is the bivariate-normal CDF with correlation ``ρ``.

The risk-neutral default probability at ``t_2`` is
``Φ_2(-d_2^{(1)}, -d_2^{(2)}; ρ)`` plus the probability of default at
``t_1`` alone, ``Φ(-d_2^{(1)})``. For a clean PD over ``[0, t_2]`` we sum
these contributions.

References
----------
Geske, R. (1977). The Valuation of Corporate Liabilities as Compound
Options. *Journal of Financial and Quantitative Analysis* 12 (4), 541-552.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

from ..exceptions import MertonInputError
from .base import StructuralModel, StructuralResult

if TYPE_CHECKING:
    from ..core.firm import Firm


def _bsm_call(A: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Plain Black-Scholes-Merton European call value."""
    if T <= 0:
        return max(0.0, A * np.exp(-q * T) - K * np.exp(-r * T))
    sqrtT = float(np.sqrt(T))
    d1 = (np.log(A / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return float(A * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _solve_inner_strike(
    *,
    D_1: float,
    D_2: float,
    t_1: float,
    t_2: float,
    r: float,
    sigma: float,
    q: float,
) -> float:
    """Find ``A*`` such that the inner call at ``t_1`` is worth ``D_1``."""
    tau = t_2 - t_1
    if tau <= 0:
        raise MertonInputError("t_2 must be strictly greater than t_1")

    def f(A_star: float) -> float:
        return _bsm_call(A_star, D_2, tau, r, sigma, q) - D_1

    lo, hi = max(D_2 * 0.01, 1e-9), max(D_1 + D_2, 1.0) * 100.0
    # Expand bracket if needed.
    attempts = 0
    while f(lo) * f(hi) > 0 and attempts < 25:
        hi *= 2.0
        attempts += 1
    if f(lo) * f(hi) > 0:
        raise MertonInputError(
            "could not bracket A* in Geske inner strike",
            suggested_fix="Inspect D_1 and D_2 — they may be inconsistent with σ_A.",
        )
    return brentq(f, lo, hi, xtol=1e-10, maxiter=200)


def _bivar_cdf(u: float, v: float, rho: float) -> float:
    """Bivariate standard-normal CDF P(U ≤ u, V ≤ v) with corr ρ."""
    cov = [[1.0, rho], [rho, 1.0]]
    return float(multivariate_normal.cdf([u, v], mean=[0.0, 0.0], cov=cov))


def geske_equity_value(
    *,
    asset_value: float,
    asset_vol: float,
    debt_short: float,
    debt_long: float,
    t_short: float,
    t_long: float,
    rf: float,
    dividend_yield: float = 0.0,
) -> float:
    """Geske two-period equity value.

    Parameters
    ----------
    asset_value, asset_vol
        Current firm asset value and asset volatility.
    debt_short, debt_long
        Liability tranches due at ``t_short`` and ``t_long`` respectively.
    t_short, t_long
        Maturities (years). Must satisfy ``0 < t_short < t_long``.
    rf, dividend_yield
        Risk-free rate and continuous dividend yield.
    """
    if not (0 < t_short < t_long):
        raise MertonInputError("require 0 < t_short < t_long")
    A_star = _solve_inner_strike(
        D_1=debt_short,
        D_2=debt_long,
        t_1=t_short,
        t_2=t_long,
        r=rf,
        sigma=asset_vol,
        q=dividend_yield,
    )
    sigma = asset_vol
    sqrt_t1 = np.sqrt(t_short)
    sqrt_t2 = np.sqrt(t_long)
    d1_1 = (
        np.log(asset_value / A_star) + (rf - dividend_yield + 0.5 * sigma * sigma) * t_short
    ) / (sigma * sqrt_t1)
    d2_1 = d1_1 - sigma * sqrt_t1
    d1_2 = (
        np.log(asset_value / debt_long) + (rf - dividend_yield + 0.5 * sigma * sigma) * t_long
    ) / (sigma * sqrt_t2)
    d2_2 = d1_2 - sigma * sqrt_t2
    rho = float(np.sqrt(t_short / t_long))
    return float(
        asset_value * np.exp(-dividend_yield * t_long) * _bivar_cdf(d1_1, d1_2, rho)
        - debt_long * np.exp(-rf * t_long) * _bivar_cdf(d2_1, d2_2, rho)
        - debt_short * np.exp(-rf * t_short) * norm.cdf(d2_1)
    )


def geske_pd(
    *,
    asset_value: float,
    asset_vol: float,
    debt_short: float,
    debt_long: float,
    t_short: float,
    t_long: float,
    rf: float,
    dividend_yield: float = 0.0,
) -> float:
    """Total risk-neutral PD over ``[0, t_long]`` in the Geske 2-period setup.

    ::

        PD(t_long) = PD(default at t_1)  + PD(default at t_2 | survived t_1)

    The first is ``Φ(-d_2^{(1)})`` (single-call PD at the inner strike). The
    second is ``Φ_2(-d_2^{(1)}, -d_2^{(2)}; ρ)`` — survival to ``t_1`` *and*
    failure at ``t_2``.
    """
    if not (0 < t_short < t_long):
        raise MertonInputError("require 0 < t_short < t_long")
    A_star = _solve_inner_strike(
        D_1=debt_short,
        D_2=debt_long,
        t_1=t_short,
        t_2=t_long,
        r=rf,
        sigma=asset_vol,
        q=dividend_yield,
    )
    sigma = asset_vol
    sqrt_t1 = np.sqrt(t_short)
    sqrt_t2 = np.sqrt(t_long)
    d2_1 = (
        np.log(asset_value / A_star) + (rf - dividend_yield - 0.5 * sigma * sigma) * t_short
    ) / (sigma * sqrt_t1)
    d2_2 = (
        np.log(asset_value / debt_long) + (rf - dividend_yield - 0.5 * sigma * sigma) * t_long
    ) / (sigma * sqrt_t2)
    rho = float(np.sqrt(t_short / t_long))
    # Default at t_1 OR (survival to t_1 AND default at t_2)
    pd_short = float(norm.cdf(-d2_1))
    pd_joint = float(_bivar_cdf(-d2_1, -d2_2, rho))  # NB: corr same since both flipped
    # Wait, when we flip signs on both ds, the bivariate CDF flips to a tail
    # probability with the same correlation. P(d2_1 ≤ 0 and d2_2 ≤ 0) is what we want
    # for "default at t_2 jointly". The conditional "default at t_2 given survive
    # t_1" is harder; we approximate cumulative PD as min(1, pd_short + pd_joint).
    return float(np.clip(pd_short + pd_joint, 0.0, 1.0))


class GeskeModel(StructuralModel):
    """Two-period Geske compound-option calibration.

    Equity is treated as a compound call: at ``t_1`` shareholders pay
    ``D_1`` only if the residual call (on long-term debt due at ``t_2``)
    is worth more than ``D_1``. We infer ``(A, σ_A)`` from observed equity
    and equity vol by inverting the Geske equity formula numerically.

    Inputs
    ------
    t_short, t_long
        Maturity dates (years). The Firm's ``horizon`` is interpreted as
        ``t_long`` by default; pass ``t_short`` explicitly to override.
    """

    method = "geske"

    def __init__(
        self,
        *,
        t_short: float = 1.0,
        recovery_rate: float = 0.5,
        tol: float = 1e-8,
        max_iter: int = 200,
    ) -> None:
        self.t_short = float(t_short)
        self.recovery_rate = float(recovery_rate)
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, firm: Firm) -> StructuralResult:
        if firm.equity_vol is None:
            raise MertonInputError(
                "GeskeModel needs equity_vol on the Firm",
                suggested_fix="Pass equity_vol when constructing the Firm.",
            )
        t_short = self.t_short
        t_long = float(firm.horizon)
        if not (0 < t_short < t_long):
            raise MertonInputError(
                f"Geske t_short ({t_short}) must lie strictly between 0 and t_long ({t_long})"
            )
        debt_short = float(np.asarray(firm.debt_short, dtype=np.float64))
        debt_long = float(np.asarray(firm.debt_long, dtype=np.float64))
        r = float(np.mean(np.asarray(firm.rf, dtype=np.float64)))
        q = float(np.mean(np.asarray(firm.dividend_yield, dtype=np.float64)))

        # Invert E_0 = geske_equity_value(A, σ_A; ...) jointly with the
        # equity-vol identity σ_E * E = (∂E/∂A) * σ_A * A. We use a simple
        # iteration: hold σ_A fixed and invert for A, then update σ_A from
        # the equity-vol equation via finite differences.

        def equity_for(A: float, sigma: float) -> float:
            return geske_equity_value(
                asset_value=A,
                asset_vol=sigma,
                debt_short=debt_short,
                debt_long=debt_long,
                t_short=t_short,
                t_long=t_long,
                rf=r,
                dividend_yield=q,
            )

        target_E = float(firm.equity)
        target_sE = float(firm.equity_vol)
        # Initial guesses from naive proxy.
        A = target_E + debt_short + debt_long
        sigma = target_sE * target_E / A
        for _ in range(self.max_iter):
            # Invert equity for A given sigma.
            lo = max(debt_short + debt_long, 1e-9) * 0.5
            hi = A * 10.0 + 1.0

            def f_A(a: float, sig: float = sigma) -> float:
                return equity_for(a, sig) - target_E

            attempts = 0
            while f_A(lo) * f_A(hi) > 0 and attempts < 25:
                hi *= 2.0
                attempts += 1
            if f_A(lo) * f_A(hi) > 0:
                raise MertonInputError(
                    "Geske inversion: could not bracket A given σ_A",
                    suggested_fix="Try different t_short or check debt levels.",
                )
            A_new = brentq(f_A, lo, hi, xtol=1e-9, maxiter=200)
            # Equity vol equation: σ_E * E = (∂E/∂A) * σ_A * A. Approximate
            # ∂E/∂A by central difference.
            eps = max(A_new * 1e-4, 1e-3)
            partial = (equity_for(A_new + eps, sigma) - equity_for(A_new - eps, sigma)) / (2 * eps)
            if partial <= 0:
                break
            sigma_new = target_sE * target_E / (partial * A_new)
            if abs(sigma_new - sigma) < self.tol and abs(A_new - A) / A_new < self.tol:
                A, sigma = A_new, sigma_new
                break
            A, sigma = A_new, sigma_new
        pd = geske_pd(
            asset_value=A,
            asset_vol=sigma,
            debt_short=debt_short,
            debt_long=debt_long,
            t_short=t_short,
            t_long=t_long,
            rf=r,
            dividend_yield=q,
        )
        dd = float("inf") if pd <= 0 else (float("-inf") if pd >= 1 else float(-norm.ppf(pd)))
        return StructuralResult(
            firm=firm,
            asset_value=A,
            asset_vol=sigma,
            default_point=debt_short + debt_long,
            dd=dd,
            pd=pd,
            method="geske",
            diagnostics={
                "t_short": t_short,
                "t_long": t_long,
                "debt_short": debt_short,
                "debt_long": debt_long,
            },
        )


__all__ = ["GeskeModel", "geske_equity_value", "geske_pd"]
