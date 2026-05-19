r"""Crosbie-Bohn / Moody's KMV iterative calibration.

Operationally identical to the Jones-Mason-Rosenfeld two-equation solver
once the **default point** is the KMV convention ``ST + 0.5·LT``. The KMV
methodology adds two practical refinements that we expose via diagnostics:

1. **Default point**: ``L = ST + 0.5 · LT`` rather than the full nominal
   debt. This is the dominant industry convention because real firms with
   long-dated maturity profiles default well before their full obligation
   comes due. The implementation reuses
   :func:`merton.core.compute_default_point` with ``kind="kmv"``.

2. **Empirical DD→EDF mapping**: KMV's commercial product converts the
   Merton distance-to-default into an *expected default frequency* (EDF)
   using a non-parametric look-up built from ~11,700 historical defaults.
   The mapping table is proprietary; we expose the hook so users can
   plug their own empirical mapping in via :attr:`KMVCalibrator.edf_map`.
   Without one, ``EDF = Φ(-DD)`` (the standard normal CDF) is reported as a
   placeholder.

References
----------
Crosbie, P. and Bohn, J. (2003). *Modeling Default Risk*. Moody's KMV.

Examples
--------
>>> from merton import Firm
>>> from merton.calibration import kmv_iterative
>>> firm = Firm(equity=100, debt_short=20, debt_long=30, equity_vol=0.30)
>>> res = kmv_iterative(
...     equity=float(firm.equity),
...     equity_vol=float(firm.equity_vol),
...     debt=float(firm.default_point_value()),
...     rf=float(firm.rf),
...     T=float(firm.horizon),
... )
>>> res.method
'kmv_iterative'
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from ..exceptions import MertonInputError
from ._solvers import solve_two_equation
from .base import CalibrationResult, Calibrator

if TYPE_CHECKING:
    from ..core.firm import Firm


EDFMap = Callable[[float], float]


def kmv_iterative(
    *,
    equity: float,
    equity_vol: float,
    debt: float,
    rf: float,
    T: float,
    dividend_yield: float = 0.0,
    edf_map: EDFMap | None = None,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> CalibrationResult:
    """Calibrate via the Crosbie-Bohn KMV procedure.

    The numerics are identical to :func:`jmr_iterative`; the difference is
    semantic: the caller is asserting that ``debt`` already encodes the
    KMV default point and that ``edf_map`` (when supplied) provides the
    empirical DD→EDF translation.
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
    # Recompute DD/EDF for diagnostics (the parent MertonResult will redo this
    # in a vectorized way, but it's handy in the calibrator's own output).
    sqrtT = float(np.sqrt(T))
    d2 = (np.log(a / debt) + (rf - dividend_yield - 0.5 * sa * sa) * T) / (sa * sqrtT)
    rn_edf = float(norm.cdf(-d2))
    empirical_edf = float(edf_map(float(d2))) if edf_map is not None else rn_edf
    return CalibrationResult(
        asset_value=a,
        asset_vol=sa,
        n_iter=n_iter,
        converged=converged,
        method="kmv_iterative",
        diagnostics={
            "default_point_convention": "KMV (ST + 0.5·LT)",
            "risk_neutral_edf": rn_edf,
            "empirical_edf": empirical_edf,
            "uses_proprietary_edf_map": edf_map is not None,
        },
    )


class KMVCalibrator(Calibrator):
    """OO wrapper around :func:`kmv_iterative`.

    Set :attr:`edf_map` to a callable ``DD -> EDF`` to use a custom empirical
    mapping. By default the risk-neutral mapping ``Φ(-DD)`` is reported.
    """

    method = "kmv_iterative"

    def __init__(
        self,
        *,
        edf_map: EDFMap | None = None,
        tol: float = 1e-8,
        max_iter: int = 200,
    ) -> None:
        self.edf_map = edf_map
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, firm: Firm) -> CalibrationResult:
        if firm.equity_vol is None:
            raise MertonInputError(
                "KMV iterative calibration requires equity_vol",
                suggested_fix="Pass equity_vol explicitly on the Firm.",
            )
        # The KMV convention is the KMV default point — enforce it gently.
        dp = firm.default_point_value()
        return kmv_iterative(
            equity=float(firm.equity),
            equity_vol=float(firm.equity_vol),
            debt=float(dp.item() if np.ndim(dp) == 0 else float(np.mean(dp))),
            rf=float(np.mean(np.asarray(firm.rf, dtype=np.float64))),
            T=float(firm.horizon),
            dividend_yield=float(np.mean(np.asarray(firm.dividend_yield, dtype=np.float64))),
            edf_map=self.edf_map,
            tol=self.tol,
            max_iter=self.max_iter,
        )


__all__ = ["EDFMap", "KMVCalibrator", "kmv_iterative"]
