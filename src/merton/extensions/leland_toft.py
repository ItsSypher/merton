r"""Leland-Toft (1996) endogenous-default model.

Leland & Toft extend Merton with **coupon-paying perpetual debt**, **taxes**,
**bankruptcy costs**, and — crucially — an **endogenous default boundary**:
equity holders choose ``V_B``, the asset-value level at which they walk
away, to maximise their own equity claim.

Setup
-----
Assets ``V_t`` follow geometric Brownian motion under the risk-neutral
measure with a continuous payout rate ``δ`` (dividends + coupon-from-assets):

.. math::

    dV/V = (r - \delta)\,dt + \sigma_A\,dW.

Debt pays a continuous coupon ``C`` per unit time and is perpetual. The
tax shield on coupons is ``\tau C`` (corporate tax rate). Default triggers
a bankruptcy cost ``\alpha`` so debt holders recover ``(1 - \alpha) V_B``.

Closed forms
------------
Define

.. math::

    x = -\frac{(r - \delta - \sigma_A^2/2) + \sqrt{(r - \delta - \sigma_A^2/2)^2 + 2 r \sigma_A^2}}{\sigma_A^2}.

Then ``(V / V_B)^{x}`` is the risk-neutral probability of eventually
defaulting (i.e. of first passage from ``V`` down to ``V_B``).

Equity value:

.. math::

    E(V) = V - (1 - \tau)\,\frac{C}{r}\left(1 - (V_B/V)^x\right)
           - V_B (V_B/V)^x.

(Equity gives up the *full* ``V_B`` at default — the bankruptcy cost
``\alpha V_B`` is destroyed, not transferred to equity, so it does not
appear here. It reduces the debt value instead.)

Equity-holders' first-order condition (the smooth-pasting condition) yields
the *optimal* default boundary

.. math::

    V_B^* = \frac{(1 - \tau)\,C\,x}{r\,(1 + x)}.

This module exposes ``optimal_default_boundary``, ``leland_toft_pd``,
``leland_toft_equity_value``, ``leland_toft_debt_value``, and
``LelandToftModel``.

References
----------
Leland, H. E. (1994). Corporate Debt Value, Bond Covenants, and Optimal
Capital Structure. *Journal of Finance* 49 (4), 1213-1252.

Leland, H. E., Toft, K. B. (1996). Optimal Capital Structure, Endogenous
Bankruptcy, and the Term Structure of Credit Spreads. *Journal of Finance*
51 (3), 987-1019.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._typing import ArrayLike, FloatArray
from ..exceptions import MertonInputError
from .base import StructuralModel, StructuralResult

if TYPE_CHECKING:
    from ..core.firm import Firm


def _x_exponent(
    rf: ArrayLike,
    sigma: ArrayLike,
    dividend_yield: ArrayLike = 0.0,
) -> FloatArray:
    r"""Compute the LT exponent ``x`` (positive root).

    The characteristic quadratic ``½σ²y² + (r - δ - σ²/2)y - r = 0`` has two
    roots; the *positive* one drives the down-and-out first-passage formula:

    .. math::

        x = \frac{\sqrt{(r - \delta - \sigma^2/2)^2 + 2 r \sigma^2}
              - (r - \delta - \sigma^2/2)}{\sigma^2}.

    With this convention ``PD = (V_B / V)^x`` for ``V > V_B``, which lies in
    ``[0, 1]`` as required.
    """
    r = np.asarray(rf, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    d = np.asarray(dividend_yield, dtype=np.float64)
    a = r - d - 0.5 * s * s
    return (np.sqrt(a * a + 2.0 * r * s * s) - a) / (s * s)


def optimal_default_boundary(
    *,
    coupon: ArrayLike,
    rf: ArrayLike,
    sigma_asset: ArrayLike,
    tax_rate: ArrayLike = 0.0,
    dividend_yield: ArrayLike = 0.0,
) -> FloatArray:
    r"""Smooth-pasting optimal default boundary ``V_B^*``."""
    r = np.asarray(rf, dtype=np.float64)
    s = np.asarray(sigma_asset, dtype=np.float64)
    C = np.asarray(coupon, dtype=np.float64)
    tau = np.asarray(tax_rate, dtype=np.float64)
    if np.any(r <= 0):
        raise MertonInputError("rf must be strictly positive")
    if np.any(s <= 0):
        raise MertonInputError("sigma_asset must be strictly positive")
    if np.any(C < 0):
        raise MertonInputError("coupon must be non-negative")
    if np.any((tau < 0) | (tau > 1)):
        raise MertonInputError("tax_rate must lie in [0, 1]")
    x = _x_exponent(r, s, dividend_yield)
    return (1.0 - tau) * C * x / (r * (1.0 + x))


def leland_toft_pd(
    *,
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    coupon: ArrayLike,
    rf: ArrayLike,
    tax_rate: ArrayLike = 0.0,
    dividend_yield: ArrayLike = 0.0,
    default_boundary: ArrayLike | None = None,
) -> FloatArray:
    r"""Risk-neutral probability of eventually defaulting under Leland-Toft.

    ``PD = (V_B / V)^x`` for ``V > V_B`` and 1 otherwise. Pass
    ``default_boundary`` to override the smooth-pasting optimum (useful
    for sensitivity analysis around a different barrier choice).
    """
    V = np.asarray(asset_value, dtype=np.float64)
    s = np.asarray(asset_vol, dtype=np.float64)
    r = np.asarray(rf, dtype=np.float64)
    if np.any(V <= 0):
        raise MertonInputError("asset_value must be strictly positive")
    if default_boundary is None:
        VB = optimal_default_boundary(
            coupon=coupon,
            rf=r,
            sigma_asset=s,
            tax_rate=tax_rate,
            dividend_yield=dividend_yield,
        )
    else:
        VB = np.asarray(default_boundary, dtype=np.float64)
        if np.any(VB <= 0):
            raise MertonInputError("default_boundary must be strictly positive")
    x = _x_exponent(r, s, dividend_yield)
    ratio = np.where(V > VB, VB / V, 1.0)
    return np.clip(np.power(ratio, x), 0.0, 1.0)


def leland_toft_equity_value(
    *,
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    coupon: ArrayLike,
    rf: ArrayLike,
    tax_rate: ArrayLike = 0.0,
    dividend_yield: ArrayLike = 0.0,
    bankruptcy_cost: ArrayLike = 0.0,
    default_boundary: ArrayLike | None = None,
) -> FloatArray:
    r"""Equity value ``E(V)`` per Leland-Toft."""
    V = np.asarray(asset_value, dtype=np.float64)
    s = np.asarray(asset_vol, dtype=np.float64)
    r = np.asarray(rf, dtype=np.float64)
    tau = np.asarray(tax_rate, dtype=np.float64)
    C = np.asarray(coupon, dtype=np.float64)
    alpha = np.asarray(bankruptcy_cost, dtype=np.float64)
    if np.any((alpha < 0) | (alpha > 1)):
        raise MertonInputError("bankruptcy_cost must lie in [0, 1]")
    _ = alpha  # α only affects the *debt* value, not equity
    if default_boundary is None:
        VB = optimal_default_boundary(
            coupon=C,
            rf=r,
            sigma_asset=s,
            tax_rate=tau,
            dividend_yield=dividend_yield,
        )
    else:
        VB = np.asarray(default_boundary, dtype=np.float64)
    x = _x_exponent(r, s, dividend_yield)
    ratio = np.where(V > VB, VB / V, 1.0)
    pd = np.power(ratio, x)
    coupon_after_tax = (1.0 - tau) * C / r
    # Equity holders give up the *full* V_B at default. The bankruptcy
    # cost α·V_B is destroyed (not transferred to equity), so it does
    # not appear here — it reduces the debt value instead. The accounting
    # identity is E + D = V + τC/r·(1-PD) - α·V_B·PD = V + TB - BC.
    return V - coupon_after_tax * (1.0 - pd) - VB * pd


def leland_toft_debt_value(
    *,
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    coupon: ArrayLike,
    rf: ArrayLike,
    tax_rate: ArrayLike = 0.0,
    dividend_yield: ArrayLike = 0.0,
    bankruptcy_cost: ArrayLike = 0.0,
    default_boundary: ArrayLike | None = None,
) -> FloatArray:
    """Debt value: present value of coupon stream + recovery at default."""
    pd = leland_toft_pd(
        asset_value=asset_value,
        asset_vol=asset_vol,
        coupon=coupon,
        rf=rf,
        tax_rate=tax_rate,
        dividend_yield=dividend_yield,
        default_boundary=default_boundary,
    )
    r = np.asarray(rf, dtype=np.float64)
    C = np.asarray(coupon, dtype=np.float64)
    alpha = np.asarray(bankruptcy_cost, dtype=np.float64)
    if default_boundary is None:
        VB = optimal_default_boundary(
            coupon=C,
            rf=r,
            sigma_asset=asset_vol,
            tax_rate=tax_rate,
            dividend_yield=dividend_yield,
        )
    else:
        VB = np.asarray(default_boundary, dtype=np.float64)
    coupon_pv = C / r * (1.0 - pd)
    recovery_pv = (1.0 - alpha) * VB * pd
    return coupon_pv + recovery_pv


class LelandToftModel(StructuralModel):
    """Calibrate Leland-Toft on a single firm with coupon-paying debt.

    Inputs
    ------
    coupon
        Continuous coupon payment (per year, in currency units). Required.
    tax_rate
        Corporate tax rate (defaults to 0).
    bankruptcy_cost
        Fraction of assets lost in bankruptcy (defaults to 0).
    """

    method = "leland_toft"

    def __init__(
        self,
        *,
        coupon: float,
        tax_rate: float = 0.0,
        bankruptcy_cost: float = 0.0,
        tol: float = 1e-8,
        max_iter: int = 200,
    ) -> None:
        self.coupon = float(coupon)
        self.tax_rate = float(tax_rate)
        self.bankruptcy_cost = float(bankruptcy_cost)
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, firm: Firm) -> StructuralResult:
        from ..calibration._solvers import solve_two_equation

        if firm.equity_vol is None:
            raise MertonInputError(
                "LelandToftModel needs equity_vol on the Firm",
                suggested_fix="Pass equity_vol explicitly.",
            )
        dp_arr = firm.default_point_value()
        debt = float(dp_arr.item() if np.ndim(dp_arr) == 0 else float(np.mean(dp_arr)))
        r = float(np.mean(np.asarray(firm.rf, dtype=np.float64)))
        q = float(np.mean(np.asarray(firm.dividend_yield, dtype=np.float64)))
        # Use the JMR two-equation solve to back out (A, σ_A) from
        # observed equity + equity vol. This is the same pre-calibration
        # step used by BlackCoxModel.
        a_val, sigma_a, _, _ = solve_two_equation(
            E=float(firm.equity),
            sigma_E=float(firm.equity_vol),
            D=debt,
            r=r,
            T=float(firm.horizon),
            q=q,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        VB = float(
            optimal_default_boundary(
                coupon=self.coupon,
                rf=r,
                sigma_asset=sigma_a,
                tax_rate=self.tax_rate,
                dividend_yield=q,
            )
        )
        pd = float(
            leland_toft_pd(
                asset_value=a_val,
                asset_vol=sigma_a,
                coupon=self.coupon,
                rf=r,
                tax_rate=self.tax_rate,
                dividend_yield=q,
                default_boundary=VB,
            )
        )
        if pd <= 0:
            dd = float("inf")
        elif pd >= 1:
            dd = float("-inf")
        else:
            from scipy.stats import norm

            dd = float(-norm.ppf(pd))
        return StructuralResult(
            firm=firm,
            asset_value=a_val,
            asset_vol=sigma_a,
            default_point=VB,
            dd=dd,
            pd=pd,
            method="leland_toft",
            diagnostics={
                "coupon": self.coupon,
                "tax_rate": self.tax_rate,
                "bankruptcy_cost": self.bankruptcy_cost,
                "optimal_default_boundary": VB,
            },
        )


__all__ = [
    "LelandToftModel",
    "leland_toft_debt_value",
    "leland_toft_equity_value",
    "leland_toft_pd",
    "optimal_default_boundary",
]
