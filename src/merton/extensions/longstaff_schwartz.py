r"""Longstaff-Schwartz (1995) two-factor structural model.

Generalises Merton by letting the **risk-free rate** evolve stochastically
under a Vasicek process. Asset value ``V_t`` follows GBM and the short
rate ``r_t`` follows mean-reverting Ornstein-Uhlenbeck:

.. math::

    dV/V &= (r_t - q)\,dt + \sigma_V\,dW^V, \\
    dr   &= \kappa(\theta - r)\,dt + \eta\,dW^r, \\
    \mathrm{Corr}(dW^V, dW^r) &= \rho.

Default occurs at the first time ``V_t`` hits a constant barrier ``K`` in
``[0, T]``. Allowing stochastic rates widens credit spreads in regimes
where ``\mathrm{Corr}(\Delta r, \Delta V) < 0`` — when rates spike,
asset values fall, pushing the firm closer to its barrier.

Implementation
--------------
Closed-form PDs only exist for special parameter combinations. The
practical path is **Monte Carlo simulation** of correlated
``(V_t, r_t)`` paths with first-passage tracking. We expose
:func:`longstaff_schwartz_pd_mc` for the MC estimator and
:class:`LongstaffSchwartzModel` for a calibrated single-firm fit.

References
----------
Longstaff, F. A., Schwartz, E. S. (1995). A Simple Approach to Valuing
Risky Fixed and Floating Rate Debt. *Journal of Finance* 50 (3), 789-819.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from ..exceptions import MertonInputError
from .base import StructuralModel, StructuralResult

if TYPE_CHECKING:
    from ..core.firm import Firm


@dataclass(slots=True, frozen=True)
class VasicekParams:
    """Vasicek short-rate parametrisation: ``dr = κ(θ - r) dt + η dW``."""

    kappa: float  # mean reversion speed
    theta: float  # long-run mean
    eta: float  # volatility
    r0: float  # initial short-rate level

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise MertonInputError("kappa must be strictly positive")
        if self.eta <= 0:
            raise MertonInputError("eta must be strictly positive")


def longstaff_schwartz_pd_mc(
    *,
    asset_value: float,
    asset_vol: float,
    barrier: float,
    T: float,
    vasicek: VasicekParams,
    correlation: float = 0.0,
    dividend_yield: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 252,
    seed: int | None = None,
) -> tuple[float, float]:
    r"""Monte Carlo first-passage PD over ``[0, T]``.

    Returns
    -------
    pd, stderr
        The MC point estimate and its standard error (≈ ``\sqrt{p(1-p)/N}``).
    """
    if asset_value <= 0:
        raise MertonInputError("asset_value must be strictly positive")
    if barrier <= 0:
        raise MertonInputError("barrier must be strictly positive")
    if asset_value <= barrier:
        return 1.0, 0.0
    if T <= 0:
        raise MertonInputError("T must be strictly positive")
    if asset_vol <= 0:
        raise MertonInputError("asset_vol must be strictly positive")
    if not -1.0 <= correlation <= 1.0:
        raise MertonInputError("correlation must lie in [-1, 1]")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = float(np.sqrt(dt))
    # Cholesky for correlated Brownian shocks.
    L = np.array([[1.0, 0.0], [correlation, np.sqrt(max(1.0 - correlation**2, 0.0))]])
    log_V = np.full(n_paths, float(np.log(asset_value)))
    r = np.full(n_paths, float(vasicek.r0))
    defaulted = np.zeros(n_paths, dtype=bool)
    log_barrier = float(np.log(barrier))
    for _ in range(n_steps):
        z = rng.standard_normal(size=(n_paths, 2))
        z_corr = z @ L.T
        dW_V = z_corr[:, 0] * sqrt_dt
        dW_r = z_corr[:, 1] * sqrt_dt
        log_V = log_V + (r - dividend_yield - 0.5 * asset_vol**2) * dt + asset_vol * dW_V
        r = r + vasicek.kappa * (vasicek.theta - r) * dt + vasicek.eta * dW_r
        defaulted |= ~defaulted & (log_V <= log_barrier)
    pd = float(defaulted.mean())
    stderr = float(np.sqrt(max(pd * (1.0 - pd), 0.0) / n_paths))
    return pd, stderr


def longstaff_schwartz_pd_analytic(
    *,
    asset_value: float,
    asset_vol: float,
    barrier: float,
    T: float,
    vasicek: VasicekParams,
    dividend_yield: float = 0.0,
) -> float:
    r"""Closed-form first-passage PD under the **zero-correlation** special
    case: rates and asset value are independent, so the Vasicek rate
    process only matters for discounting (not for the default trigger).
    This reduces to the standard Black-Cox formula with a deterministic
    drift equal to the unconditional mean of ``r_t``.
    """
    if not vasicek.kappa > 0:
        raise MertonInputError("Vasicek κ must be > 0")
    # Mean of integrated rate over [0, T] is θ + (r0 - θ)(1 - e^{-κT})/(κT).
    mean_r = vasicek.theta + (vasicek.r0 - vasicek.theta) * (
        1.0 - np.exp(-vasicek.kappa * T)
    ) / max(vasicek.kappa * T, 1e-12)
    a = float(np.log(asset_value / barrier))
    nu = mean_r - dividend_yield - 0.5 * asset_vol**2
    sqrtT = float(np.sqrt(T))
    term1 = norm.cdf((-a - nu * T) / (asset_vol * sqrtT))
    expo = np.clip(-2.0 * nu * a / (asset_vol**2), -700.0, 700.0)
    term2 = np.exp(expo) * norm.cdf((-a + nu * T) / (asset_vol * sqrtT))
    return float(np.clip(term1 + term2, 0.0, 1.0))


class LongstaffSchwartzModel(StructuralModel):
    """Calibrated single-firm Longstaff-Schwartz model.

    Inputs the user supplies on top of the :class:`Firm`:

    - ``vasicek`` — short-rate parametrisation.
    - ``correlation`` — between asset and rate shocks.
    - ``mc_paths``, ``mc_steps``, ``mc_seed`` — Monte Carlo controls.
    """

    method = "longstaff_schwartz"

    def __init__(
        self,
        *,
        vasicek: VasicekParams,
        correlation: float = 0.0,
        mc_paths: int = 20_000,
        mc_steps: int = 252,
        mc_seed: int | None = None,
        tol: float = 1e-8,
        max_iter: int = 200,
    ) -> None:
        self.vasicek = vasicek
        self.correlation = float(correlation)
        self.mc_paths = int(mc_paths)
        self.mc_steps = int(mc_steps)
        self.mc_seed = mc_seed
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, firm: Firm) -> StructuralResult:
        from ..calibration._solvers import solve_two_equation

        if firm.equity_vol is None:
            raise MertonInputError(
                "LongstaffSchwartzModel needs equity_vol on the Firm",
                suggested_fix="Pass equity_vol explicitly.",
            )
        dp_arr = firm.default_point_value()
        debt = float(dp_arr.item() if np.ndim(dp_arr) == 0 else float(np.mean(dp_arr)))
        r0 = float(np.mean(np.asarray(firm.rf, dtype=np.float64)))
        q = float(np.mean(np.asarray(firm.dividend_yield, dtype=np.float64)))
        a_val, sigma_a, _, _ = solve_two_equation(
            E=float(firm.equity),
            sigma_E=float(firm.equity_vol),
            D=debt,
            r=r0,
            T=float(firm.horizon),
            q=q,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        if abs(self.correlation) < 1e-6:
            pd = longstaff_schwartz_pd_analytic(
                asset_value=a_val,
                asset_vol=sigma_a,
                barrier=debt,
                T=float(firm.horizon),
                vasicek=self.vasicek,
                dividend_yield=q,
            )
            stderr = None
        else:
            pd, stderr = longstaff_schwartz_pd_mc(
                asset_value=a_val,
                asset_vol=sigma_a,
                barrier=debt,
                T=float(firm.horizon),
                vasicek=self.vasicek,
                correlation=self.correlation,
                dividend_yield=q,
                n_paths=self.mc_paths,
                n_steps=self.mc_steps,
                seed=self.mc_seed,
            )
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
            method="longstaff_schwartz",
            diagnostics={
                "vasicek": {
                    "kappa": self.vasicek.kappa,
                    "theta": self.vasicek.theta,
                    "eta": self.vasicek.eta,
                    "r0": self.vasicek.r0,
                },
                "correlation": self.correlation,
                "mc_stderr": stderr,
            },
        )


__all__ = [
    "LongstaffSchwartzModel",
    "VasicekParams",
    "longstaff_schwartz_pd_analytic",
    "longstaff_schwartz_pd_mc",
]
