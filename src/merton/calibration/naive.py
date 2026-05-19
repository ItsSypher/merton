"""Naive calibration (Bharath-Shumway 2008).

Closed-form approximation that skips the two-equation calibration entirely:

::

    A   ≈ E + D
    σ_D ≈ 0.05 + 0.25 · σ_E                 (proxy for debt volatility)
    σ_A ≈ (E / (E + D)) · σ_E + (D / (E + D)) · σ_D

Empirically (Bharath-Shumway 2008, *RFS*), this estimator forecasts default
about as well as the full Merton solution while costing essentially nothing
to evaluate.

References
----------
Bharath & Shumway (2008). *Forecasting Default with the Merton Distance to
Default Model*. Review of Financial Studies 21 (3), 1339-1369.
"""

from __future__ import annotations

import numpy as np

from .._typing import ArrayLike
from ..exceptions import MertonInputError
from .base import CalibrationResult, Calibrator


def naive(
    equity: ArrayLike,
    equity_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
) -> CalibrationResult:
    """Closed-form Bharath-Shumway naive estimator.

    Parameters
    ----------
    equity, debt
        Market value of equity and the default threshold.
    equity_vol
        Annualised equity volatility.
    rf, T
        Risk-free rate and horizon (used for the parent
        :class:`~merton.core.MertonResult` only — not inside the naive math).
    """
    E = np.asarray(equity, dtype=np.float64)
    D = np.asarray(debt, dtype=np.float64)
    se = np.asarray(equity_vol, dtype=np.float64)
    if np.any(se <= 0):
        raise MertonInputError("equity_vol must be strictly positive")
    A = E + D
    debt_vol = 0.05 + 0.25 * se
    sa = (E / A) * se + (D / A) * debt_vol
    return CalibrationResult(
        asset_value=A,
        asset_vol=float(np.asarray(sa).item() if sa.ndim == 0 else float(sa.mean())),
        n_iter=0,
        converged=True,
        method="naive",
        diagnostics={"debt_vol_proxy": float(np.mean(debt_vol))},
    )


class NaiveCalibrator(Calibrator):
    """OO interface to :func:`naive`."""

    method = "naive"

    def fit(self, firm):  # type: ignore[no-untyped-def]
        if firm.equity_vol is None:
            raise MertonInputError(
                "naive calibration requires equity_vol",
                suggested_fix="Pass equity_vol on the Firm or use a different method.",
            )
        debt = firm.default_point_value()
        return naive(
            equity=firm.equity,
            equity_vol=firm.equity_vol,
            debt=debt,
            rf=firm.rf,
            T=firm.horizon,
        )


__all__ = ["NaiveCalibrator", "naive"]
