"""Jones-Mason-Rosenfeld (1984) iterative calibration.

Solve the two simultaneous equations

::

    E    = A · Φ(d₁) - D · e^{-rT} · Φ(d₂)              (BSM call value)
    σ_E  = (A / E) · Φ(d₁) · σ_A                         (Ito's lemma)

for the unknowns ``A`` (asset value) and ``σ_A`` (asset volatility).
Practically identical to the proprietary Crosbie-Bohn KMV iteration when
applied to a snapshot of one firm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..exceptions import MertonInputError
from ._solvers import solve_two_equation
from .base import CalibrationResult, Calibrator

if TYPE_CHECKING:
    from ..core.firm import Firm


def jmr_iterative(
    *,
    equity: float,
    equity_vol: float,
    debt: float,
    rf: float,
    T: float,
    dividend_yield: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> CalibrationResult:
    """Solve the JMR two-equation system. Scalar inputs only.

    Examples
    --------
    >>> res = jmr_iterative(equity=80, equity_vol=0.30, debt=60, rf=0.04, T=1.0)
    >>> abs(res.asset_value - 137) < 5  # rough sanity, exact value depends on σ_E
    True
    """
    a, sa, n_iter, converged = solve_two_equation(
        E=float(equity),
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
        converged=converged,
        method="jmr_iterative",
        diagnostics={"tol": tol, "max_iter": max_iter},
    )


class JMRCalibrator(Calibrator):
    """OO wrapper around :func:`jmr_iterative`."""

    method = "jmr_iterative"

    def __init__(self, *, tol: float = 1e-8, max_iter: int = 200) -> None:
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, firm: Firm) -> CalibrationResult:
        if firm.equity_vol is None:
            raise MertonInputError(
                "JMR iterative calibration requires equity_vol",
                suggested_fix="Pass equity_vol explicitly on the Firm.",
            )
        return jmr_iterative(
            equity=float(firm.equity),
            equity_vol=float(firm.equity_vol),
            debt=float(firm.default_point_value()),
            rf=float(firm.rf),
            T=float(firm.horizon),
            dividend_yield=float(firm.dividend_yield),
            tol=self.tol,
            max_iter=self.max_iter,
        )


__all__ = ["JMRCalibrator", "jmr_iterative"]
