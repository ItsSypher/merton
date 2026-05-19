r"""Black-Cox (1976) first-passage default model.

Where Merton lets the firm default only at the debt-maturity horizon ``T``,
Black-Cox lets the firm default at the *first time* the asset value crosses a
barrier ``K(t)``. The barrier is typically constant (``K(t) = K``) or
exponentially decaying (``K(t) = K · e^{-γ(T-t)}``); both have closed-form
first-passage probabilities.

Mathematics
-----------
Let ``X_t = log(A_t)``. Under the risk-neutral measure ``X`` is a Brownian
motion with drift ``ν = r - q - σ_A²/2`` and volatility ``σ_A``. Define
``a = log(A_0 / K) > 0``. For the standard constant-barrier case, the
first-passage probability that ``X`` drops below ``log K`` somewhere in
``[0, T]`` is

.. math::

    \mathrm{PD}_{BC}
    = \Phi\!\left(\frac{-a - \nu T}{\sigma_A \sqrt{T}}\right)
      + \exp\!\left(-\frac{2 \nu a}{\sigma_A^2}\right)\,
        \Phi\!\left(\frac{-a + \nu T}{\sigma_A \sqrt{T}}\right).

When ``γ > 0`` the barrier ``K(t) = K_T · e^{-γ(T-t)}`` grows over time toward
``K_T``. A change of variables ``Y_t = log(A_t / K(t))`` reduces this to a
constant-barrier first-passage problem with shifted drift ``ν - γ``; the
formula above applies with that substitution.

References
----------
Black, F. and Cox, J. C. (1976). Valuing corporate securities: Some effects of
bond indenture provisions. *Journal of Finance* 31 (2), 351-367.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from .._typing import ArrayLike, FloatArray
from ..exceptions import MertonInputError
from .base import StructuralModel, StructuralResult

if TYPE_CHECKING:
    from ..core.firm import Firm


@dataclass(slots=True)
class _BCParams:
    asset_value: float
    asset_vol: float
    debt: float
    rf: float
    T: float
    dividend_yield: float = 0.0
    barrier_growth_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.asset_value <= 0:
            raise MertonInputError("asset_value must be strictly positive")
        if self.asset_vol <= 0:
            raise MertonInputError("asset_vol must be strictly positive")
        if self.debt <= 0:
            raise MertonInputError("debt must be strictly positive")
        if self.T <= 0:
            raise MertonInputError("T must be strictly positive")


def black_cox_pd(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
    *,
    dividend_yield: ArrayLike = 0.0,
    barrier_growth_rate: ArrayLike = 0.0,
) -> FloatArray:
    r"""Closed-form Black-Cox risk-neutral PD.

    Parameters
    ----------
    asset_value, asset_vol, debt, rf, T
        Standard structural-model inputs.
    dividend_yield
        Continuous dividend yield ``q`` (subtracted from the drift).
    barrier_growth_rate
        ``γ`` in the time-dependent barrier ``K(t) = K_T · e^{-γ(T-t)}``.
        ``γ = 0`` (the default) corresponds to a constant barrier.

    Returns
    -------
    FloatArray
        Risk-neutral probability that the asset value touches ``K(t)``
        somewhere on ``[0, T]``.
    """
    A = np.asarray(asset_value, dtype=np.float64)
    s = np.asarray(asset_vol, dtype=np.float64)
    D = np.asarray(debt, dtype=np.float64)
    r = np.asarray(rf, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    q = np.asarray(dividend_yield, dtype=np.float64)
    gamma = np.asarray(barrier_growth_rate, dtype=np.float64)

    if np.any(A <= 0) or np.any(s <= 0) or np.any(D <= 0) or np.any(T_ <= 0):
        raise MertonInputError("asset_value, asset_vol, debt, T must be strictly positive")
    a = np.log(A / D)
    nu = r - q - 0.5 * s * s - gamma  # shifted drift
    sqrtT = np.sqrt(T_)
    term1 = norm.cdf((-a - nu * T_) / (s * sqrtT))
    # exp(-2 ν a / σ²). a > 0 by construction (a < 0 would mean A < D, but we
    # would still return a valid value in [0, 1]; clip to avoid overflow).
    expo = -2.0 * nu * a / (s * s)
    term2 = np.exp(np.clip(expo, -700.0, 700.0)) * norm.cdf((-a + nu * T_) / (s * sqrtT))
    pd = term1 + term2
    return np.clip(pd, 0.0, 1.0)


def black_cox_survival(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
    *,
    dividend_yield: ArrayLike = 0.0,
    barrier_growth_rate: ArrayLike = 0.0,
) -> FloatArray:
    """``1 - black_cox_pd(...)``."""
    return 1.0 - black_cox_pd(
        asset_value,
        asset_vol,
        debt,
        rf,
        T,
        dividend_yield=dividend_yield,
        barrier_growth_rate=barrier_growth_rate,
    )


class BlackCoxModel(StructuralModel):
    """Calibrate Black-Cox on a single firm.

    The model still needs the asset value ``A`` and asset volatility
    ``σ_A``; we obtain them via the same JMR two-equation system used by
    Merton (taking the user's equity_vol as the input). For real
    practitioners with credit-spread data, joint calibration to bonds /
    CDS is the future-phase route.
    """

    method = "black_cox"

    def __init__(
        self,
        *,
        barrier_growth_rate: float = 0.0,
        recovery_rate: float = 0.5,
        tol: float = 1e-8,
        max_iter: int = 200,
    ) -> None:
        self.barrier_growth_rate = float(barrier_growth_rate)
        self.recovery_rate = float(recovery_rate)
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, firm: Firm) -> StructuralResult:
        from ..calibration._solvers import solve_two_equation

        if firm.equity_vol is None:
            raise MertonInputError(
                "BlackCoxModel needs equity_vol on the Firm",
                suggested_fix="Pass equity_vol when constructing the Firm.",
            )
        dp_arr = firm.default_point_value()
        debt = float(dp_arr.item() if np.ndim(dp_arr) == 0 else float(np.mean(dp_arr)))
        r = float(np.mean(np.asarray(firm.rf, dtype=np.float64)))
        q = float(np.mean(np.asarray(firm.dividend_yield, dtype=np.float64)))

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
        pd = float(
            black_cox_pd(
                a_val,
                sigma_a,
                debt,
                r,
                firm.horizon,
                dividend_yield=q,
                barrier_growth_rate=self.barrier_growth_rate,
            )
        )
        # DD reported via the inverse-normal of survival, so a higher DD ⇒
        # smaller PD (matches the Merton convention).
        if pd <= 0:
            dd = float("inf")
        elif pd >= 1:
            dd = float("-inf")
        else:
            dd = float(-norm.ppf(pd))
        return StructuralResult(
            firm=firm,
            asset_value=a_val,
            asset_vol=sigma_a,
            default_point=debt,
            dd=dd,
            pd=pd,
            method="black_cox",
            diagnostics={
                "barrier_growth_rate": self.barrier_growth_rate,
                "recovery_rate": self.recovery_rate,
            },
        )


__all__ = ["BlackCoxModel", "black_cox_pd", "black_cox_survival"]
