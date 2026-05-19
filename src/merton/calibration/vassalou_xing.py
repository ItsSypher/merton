"""Vassalou-Xing (2004) iterative MLE calibration.

When supplied a *single snapshot* of equity value + equity volatility, this
calibrator coincides with :func:`jmr_iterative`.

When supplied an equity *time series*, it runs the canonical Vassalou-Xing
loop:

1. Initialise ``σ_A^(0) = σ_E · E / (E + D)`` (naive proxy).
2. Given ``σ_A^(k)``, invert the BSM call equation at each ``t`` to obtain
   ``A_t`` from the observed ``E_t``.
3. Compute log-returns ``r_{A,t} = ln(A_t / A_{t-1})`` and update
   ``σ_A^(k+1) = std(r_A) · √252``.
4. Repeat until ``|σ_A^(k+1) - σ_A^(k)| < tol`` or ``k == max_iter``.

References
----------
Vassalou & Xing (2004). *Default Risk in Equity Returns*. Journal of
Finance 59 (2), 831-868.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from .._backend._numpy import equity_value as _equity_value_kernel
from .._typing import FloatArray
from ..exceptions import (
    CalibrationConvergenceError,
    InsufficientDataError,
    MertonInputError,
)
from ._solvers import solve_two_equation
from .base import CalibrationResult, Calibrator

if TYPE_CHECKING:
    from ..core.firm import Firm

ANNUALIZATION = 252.0


def _invert_equity_for_assets(
    equity_series: FloatArray,
    sigma_A: float,
    debt: float,
    rf: float,
    T: float,
    q: float,
) -> FloatArray:
    """Solve E = BSCall(A; σ_A, D, r, T) for ``A`` at each ``t``."""
    A_series = np.empty_like(equity_series, dtype=np.float64)
    for i, E in enumerate(equity_series):

        def f(A: float, E_val: float = float(E)) -> float:
            return float(_equity_value_kernel(A, sigma_A, debt, rf, T, q) - E_val)

        # Bracket: A must exceed the discounted debt at minimum and lie below
        # several multiples of equity + debt at maximum.
        lo = max(E + debt * np.exp(-rf * T), 1e-9) * 0.5
        hi = (E + debt) * 5.0
        # Expand brackets if f doesn't change sign.
        f_lo, f_hi = f(lo), f(hi)
        attempts = 0
        while f_lo * f_hi > 0 and attempts < 20:
            hi *= 2.0
            f_hi = f(hi)
            attempts += 1
        if f_lo * f_hi > 0:
            raise CalibrationConvergenceError(
                "could not bracket asset value during VX inversion",
                suggested_fix="Inspect equity series for outliers / zeros.",
            )
        A_series[i] = brentq(f, lo, hi, xtol=1e-10)
    return A_series


def vassalou_xing(
    *,
    equity: float | FloatArray,
    debt: float,
    rf: float,
    T: float,
    equity_vol: float | None = None,
    dividend_yield: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 200,
    annualization: float = ANNUALIZATION,
) -> CalibrationResult:
    """Vassalou-Xing iterative MLE calibration.

    Either pass:

    - ``equity`` as a scalar with ``equity_vol`` — this collapses to the JMR
      two-equation solve.
    - ``equity`` as a 1-D array (equity series) — full iterative VX. In this
      case ``equity_vol`` is optional; if omitted we estimate it from the
      sample stdev of equity log-returns.

    Returns
    -------
    CalibrationResult
        For the array case, ``asset_value`` is a full array of inferred
        ``A_t`` and ``asset_vol`` is the converged scalar ``σ_A``.
    """
    eq_arr = np.atleast_1d(np.asarray(equity, dtype=np.float64))
    if eq_arr.ndim != 1:
        raise MertonInputError("equity must be scalar or 1-D")
    if eq_arr.size == 1:
        # Single snapshot ⇒ delegate to JMR.
        if equity_vol is None:
            raise MertonInputError(
                "scalar Vassalou-Xing requires equity_vol",
                suggested_fix="Pass equity_vol, or provide an equity time series.",
            )
        a, sa, n_iter, conv = solve_two_equation(
            E=float(eq_arr[0]),
            sigma_E=float(equity_vol),
            D=float(debt),
            r=float(rf),
            T=float(T),
            q=float(dividend_yield),
            tol=tol,
            max_iter=max_iter,
        )
        return CalibrationResult(
            asset_value=a,
            asset_vol=sa,
            n_iter=n_iter,
            converged=conv,
            method="vassalou_xing",
            diagnostics={"mode": "snapshot"},
        )

    # Series mode.
    if eq_arr.size < 10:
        raise InsufficientDataError(
            "Vassalou-Xing series mode needs at least 10 observations",
            suggested_fix="Use a longer equity history or method='naive'.",
        )

    eq_log_returns = np.diff(np.log(eq_arr))
    sigma_E0 = (
        float(equity_vol)
        if equity_vol is not None
        else float(eq_log_returns.std(ddof=1) * np.sqrt(annualization))
    )
    sigma_A = sigma_E0 * float(eq_arr[-1]) / (float(eq_arr[-1]) + float(debt))
    history: list[float] = [sigma_A]
    converged = False
    for k in range(max_iter):
        A_series = _invert_equity_for_assets(eq_arr, sigma_A, debt, rf, T, dividend_yield)
        ret_A = np.diff(np.log(A_series))
        sigma_A_new = float(ret_A.std(ddof=1) * np.sqrt(annualization))
        if not np.isfinite(sigma_A_new) or sigma_A_new <= 0:
            raise CalibrationConvergenceError(
                f"VX iteration produced non-finite σ_A at step {k}",
            )
        history.append(sigma_A_new)
        if abs(sigma_A_new - sigma_A) < tol:
            sigma_A = sigma_A_new
            converged = True
            break
        sigma_A = sigma_A_new
    if not converged:
        raise CalibrationConvergenceError(
            f"Vassalou-Xing did not converge in {max_iter} iterations",
            suggested_fix=f"Relax tol (now {tol}) or increase max_iter.",
        )
    A_series = _invert_equity_for_assets(eq_arr, sigma_A, debt, rf, T, dividend_yield)
    asset_drift = float(np.mean(np.diff(np.log(A_series))) * annualization)
    return CalibrationResult(
        asset_value=A_series,
        asset_vol=sigma_A,
        asset_drift=asset_drift,
        n_iter=len(history) - 1,
        converged=True,
        method="vassalou_xing",
        diagnostics={
            "mode": "series",
            "history": history,
            "sigma_E_used": sigma_E0,
            "n_obs": int(eq_arr.size),
        },
    )


class VassalouXingCalibrator(Calibrator):
    """OO wrapper around :func:`vassalou_xing`."""

    method = "vassalou_xing"

    def __init__(self, *, tol: float = 1e-6, max_iter: int = 200) -> None:
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, firm: Firm) -> CalibrationResult:
        equity = np.asarray(firm.equity, dtype=np.float64)
        debt = (
            float(np.asarray(firm.default_point_value()).item())
            if np.ndim(firm.default_point_value()) == 0
            else float(np.mean(firm.default_point_value()))
        )
        return vassalou_xing(
            equity=equity if equity.ndim > 0 and equity.size > 1 else float(equity),
            equity_vol=float(firm.equity_vol) if firm.equity_vol is not None else None,
            debt=debt,
            rf=float(np.mean(np.asarray(firm.rf, dtype=np.float64))),
            T=float(firm.horizon),
            dividend_yield=float(np.mean(np.asarray(firm.dividend_yield, dtype=np.float64))),
            tol=self.tol,
            max_iter=self.max_iter,
        )


__all__ = ["VassalouXingCalibrator", "vassalou_xing"]
