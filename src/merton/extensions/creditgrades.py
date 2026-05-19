r"""CreditGrades (Finger, Finkelstein, Pan, Lardy, Ta & Tierney, 2002).

The CreditGrades model is the RiskMetrics open-standard structural credit
model. It generalises Merton by:

1. **Stochastic default barrier** — the recovery ratio ``L`` is log-normally
   distributed (``log L = log L_bar − λ²/2 + λZ``, ``Z ~ N(0, 1)``), so the
   barrier itself is uncertain. This widens short-horizon credit spreads
   relative to Merton.
2. **Reference firm value** — ``V_0 = S_0 + L_bar · D`` per share, where
   ``S`` is equity, ``D`` is debt per share, and ``L_bar`` is the mean
   recovery ratio.
3. **Asset volatility from equity volatility** — ``σ = σ_E · S/(S + L_bar·D)``,
   the standard leverage-adjusted formula.

Closed-form survival probability (the practical approximation used in the
tech doc):

.. math::

    P(t) = \Phi\!\left(-\frac{A_t}{2} + \frac{\log d}{A_t}\right)
         - d\,\Phi\!\left(-\frac{A_t}{2} - \frac{\log d}{A_t}\right),

with

.. math::

    A_t^2 = \sigma^2 t + \lambda^2,\qquad d = \frac{V_0}{L\,D}\,e^{\lambda^2}.

The implied flat hazard rate is ``h(t) = -\log P(t) / t``; the CDS spread
is ``h(t) · LGD``.

References
----------
Finger, C. C., Finkelstein, V., Pan, G., Lardy, J.-P., Ta, T., Tierney, J.
(2002). *CreditGrades Technical Document*. RiskMetrics Group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from .._typing import ArrayLike, FloatArray
from ..exceptions import MertonInputError
from .base import StructuralModel, StructuralResult

if TYPE_CHECKING:
    from ..core.firm import Firm


# Default parameters from the 2002 technical document (Table 1).
DEFAULT_LBAR = 0.5
DEFAULT_LAMBDA = 0.3


def _validate(
    equity: ArrayLike,
    equity_vol: ArrayLike,
    debt_per_share: ArrayLike,
    T: ArrayLike,
    lbar: float,
    lam: float,
) -> None:
    S = np.asarray(equity, dtype=np.float64)
    sE = np.asarray(equity_vol, dtype=np.float64)
    D = np.asarray(debt_per_share, dtype=np.float64)
    Tarr = np.asarray(T, dtype=np.float64)
    if np.any(S <= 0):
        raise MertonInputError("equity (S) must be strictly positive")
    if np.any(sE <= 0):
        raise MertonInputError("equity_vol must be strictly positive")
    if np.any(D <= 0):
        raise MertonInputError("debt_per_share must be strictly positive")
    if np.any(Tarr <= 0):
        raise MertonInputError("T must be strictly positive")
    if not 0 < lbar <= 1:
        raise MertonInputError("lbar must lie in (0, 1]")
    if lam <= 0:
        raise MertonInputError("lambda (barrier uncertainty) must be positive")


def creditgrades_survival(
    equity: ArrayLike,
    equity_vol: ArrayLike,
    debt_per_share: ArrayLike,
    T: ArrayLike,
    *,
    lbar: float = DEFAULT_LBAR,
    lam: float = DEFAULT_LAMBDA,
) -> FloatArray:
    """Survival probability ``P(t)`` under CreditGrades.

    Parameters
    ----------
    equity, equity_vol, debt_per_share
        Equity price ``S``, annualised equity volatility ``σ_E``, and
        debt-per-share ``D``. All in the same currency units.
    T
        Horizon (years).
    lbar
        Mean recovery ratio (default 0.5).
    lam
        Standard deviation of ``log L`` (default 0.3).
    """
    _validate(equity, equity_vol, debt_per_share, T, lbar, lam)
    S = np.asarray(equity, dtype=np.float64)
    sE = np.asarray(equity_vol, dtype=np.float64)
    D = np.asarray(debt_per_share, dtype=np.float64)
    Tarr = np.asarray(T, dtype=np.float64)

    V0 = S + lbar * D
    sigma = sE * S / V0  # leverage-adjusted asset vol
    A2 = sigma * sigma * Tarr + lam * lam
    A = np.sqrt(A2)
    d = (V0 / (lbar * D)) * np.exp(lam * lam)
    logd = np.log(d)

    term1 = norm.cdf(-A / 2.0 + logd / A)
    term2 = d * norm.cdf(-A / 2.0 - logd / A)
    return np.clip(term1 - term2, 0.0, 1.0)


def creditgrades_pd(
    equity: ArrayLike,
    equity_vol: ArrayLike,
    debt_per_share: ArrayLike,
    T: ArrayLike,
    *,
    lbar: float = DEFAULT_LBAR,
    lam: float = DEFAULT_LAMBDA,
) -> FloatArray:
    """``1 - creditgrades_survival(...)`` — risk-neutral PD over ``[0, T]``."""
    return 1.0 - creditgrades_survival(equity, equity_vol, debt_per_share, T, lbar=lbar, lam=lam)


def creditgrades_spread(
    equity: ArrayLike,
    equity_vol: ArrayLike,
    debt_per_share: ArrayLike,
    T: ArrayLike,
    *,
    lgd: ArrayLike = 0.6,
    lbar: float = DEFAULT_LBAR,
    lam: float = DEFAULT_LAMBDA,
    in_bps: bool = True,
) -> FloatArray:
    """CreditGrades flat-hazard CDS spread.

    Returns ``-log(P(t))/t · LGD`` in basis points by default.
    """
    surv = creditgrades_survival(equity, equity_vol, debt_per_share, T, lbar=lbar, lam=lam)
    surv = np.clip(surv, 1e-15, 1.0)
    Tarr = np.asarray(T, dtype=np.float64)
    lgd_arr = np.asarray(lgd, dtype=np.float64)
    spread = -np.log(surv) / Tarr * lgd_arr
    return spread * 10_000.0 if in_bps else spread


class CreditGradesModel(StructuralModel):
    """CreditGrades-style structural credit model.

    The model takes equity per share, equity volatility, and debt per
    share (all in the same currency units). For aggregate-value firms,
    convert to per-share by dividing by ``shares_outstanding``.
    """

    method = "creditgrades"

    def __init__(
        self,
        *,
        lbar: float = DEFAULT_LBAR,
        lam: float = DEFAULT_LAMBDA,
        debt_per_share: float | None = None,
    ) -> None:
        self.lbar = float(lbar)
        self.lam = float(lam)
        self.debt_per_share = debt_per_share

    def fit(self, firm: Firm) -> StructuralResult:
        if firm.equity_vol is None:
            raise MertonInputError(
                "CreditGradesModel needs equity_vol on the Firm",
                suggested_fix="Pass equity_vol when constructing the Firm.",
            )
        equity = float(np.asarray(firm.equity).item())
        sE = float(firm.equity_vol)
        if self.debt_per_share is not None:
            D = float(self.debt_per_share)
        else:
            # Approximate: total debt per "unit" share. Caller can override.
            D = float(np.asarray(firm.default_point_value()).item())
        T = float(firm.horizon)
        surv = float(creditgrades_survival(equity, sE, D, T, lbar=self.lbar, lam=self.lam))
        pd = float(1.0 - surv)
        # Map PD → DD as -Φ⁻¹(PD) for a Merton-comparable scale.
        if pd <= 0:
            dd = float("inf")
        elif pd >= 1:
            dd = float("-inf")
        else:
            dd = float(-norm.ppf(pd))
        sigma_asset = sE * equity / (equity + self.lbar * D)
        return StructuralResult(
            firm=firm,
            asset_value=equity + self.lbar * D,
            asset_vol=sigma_asset,
            default_point=self.lbar * D,
            dd=dd,
            pd=pd,
            method="creditgrades",
            diagnostics={
                "lbar": self.lbar,
                "lambda": self.lam,
                "debt_per_share": D,
                "implied_cds_spread_bps": float(
                    creditgrades_spread(equity, sE, D, T, lbar=self.lbar, lam=self.lam)
                ),
            },
        )


__all__ = [
    "DEFAULT_LAMBDA",
    "DEFAULT_LBAR",
    "CreditGradesModel",
    "creditgrades_pd",
    "creditgrades_spread",
    "creditgrades_survival",
]
