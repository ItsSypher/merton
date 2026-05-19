r"""Duan (1994) transformed-data maximum-likelihood calibration.

The Duan MLE jointly estimates the asset drift ``μ`` and asset volatility
``σ_A`` directly from an *equity* time series. Unlike Vassalou-Xing's
iterative substitution, it writes the likelihood of the *equity* observations
as a function of the parameters and maximises that likelihood with a generic
optimiser.

Mathematics
-----------
Suppose we observe ``E_0, E_1, …, E_n`` at intervals ``Δt = 1/252`` and the
firm has a single zero-coupon liability ``D`` maturing at ``T`` years out.
The Black-Scholes-Merton equity-value map ``E = h(A; σ_A, D, r, T)`` is
strictly increasing in ``A``; let ``A_t = h^{-1}(E_t; σ_A, D, r, T_t)`` with
``T_t = T - t·Δt``. ``A_t`` is then geometric Brownian motion with drift
``μ`` and volatility ``σ_A`` under the physical measure. The transformed-data
log-likelihood is

.. math::

    \ell(\mu, \sigma_A)
    = \sum_{t=1}^n \log f(r_t; \mu - \sigma_A^2/2, \sigma_A^2 \Delta t)
      - \sum_{t=1}^n \log \!\bigl[A_t \Phi(d_1^t)\bigr]

where ``r_t = log(A_t / A_{t-1})`` and the second sum is the Jacobian of the
``E ↦ A`` transformation. Maximising ``\ell`` over ``(μ, σ_A)`` gives the
MLE.

Survivorship-bias correction
----------------------------
For samples that only include firms surviving to ``T_obs = n·Δt``,
Duan-Gauthier-Simonato-Zaanoun (2004) recommend dividing the likelihood by
the survival probability:

.. math::

    \ell_{corr}(\mu, \sigma_A) = \ell(\mu, \sigma_A)
                                 - \log P\!\bigl(\min_{0 \le t \le T_{obs}} A_t > D\bigr).

The survival probability has a closed-form via the reflection principle for
geometric Brownian motion (see :mod:`merton._backend._survival`).

References
----------
Duan, J.-C. (1994). Maximum Likelihood Estimation Using Price Data of the
Derivative Contract. *Mathematical Finance* 4 (2), 155-167.

Duan, J.-C., Gauthier, G., Simonato, J.-G., Zaanoun, S. (2004). Estimating
Merton's Model by Maximum Likelihood with Survivorship Consideration.
*HEC Montreal* Working Paper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq, minimize

from .._backend._numpy import equity_value as _equity_value_kernel
from .._backend._numpy import norm_cdf
from .._backend._survival import log_survival_probability
from .._typing import FloatArray
from ..exceptions import (
    CalibrationConvergenceError,
    InsufficientDataError,
    MertonInputError,
)
from .base import CalibrationResult, Calibrator

if TYPE_CHECKING:
    from ..core.firm import Firm

ANNUALIZATION = 252.0
_MIN_SIGMA = 1e-4
_MAX_SIGMA = 5.0
_HESSIAN_EPS = 1e-5


def _invert_assets(
    equity_series: FloatArray,
    sigma_A: float,
    debt: float,
    rf: float,
    T: float,
    q: float,
) -> FloatArray:
    """Solve ``E_t = BSCall(A_t; σ_A, D, r, T)`` for each ``t`` (same as VX)."""
    A_out = np.empty_like(equity_series, dtype=np.float64)
    for i, E in enumerate(equity_series):

        def f(A: float, E_val: float = float(E)) -> float:
            return float(_equity_value_kernel(A, sigma_A, debt, rf, T, q) - E_val)

        lo = max(float(E) + debt * np.exp(-rf * T), 1e-9) * 0.5
        hi = (float(E) + debt) * 5.0
        attempts = 0
        while f(lo) * f(hi) > 0 and attempts < 25:
            hi *= 2.0
            attempts += 1
        if f(lo) * f(hi) > 0:
            raise CalibrationConvergenceError(
                "could not bracket asset value during Duan MLE inversion",
                suggested_fix="Inspect equity series for zeros or sharp jumps.",
            )
        A_out[i] = brentq(f, lo, hi, xtol=1e-10, maxiter=200)
    return A_out


def _neg_log_likelihood(
    params: np.ndarray,
    *,
    equity_series: FloatArray,
    debt: float,
    rf: float,
    T: float,
    dt: float,
    q: float,
    survivor_bias_correction: bool,
) -> float:
    mu, sigma = float(params[0]), float(params[1])
    if not (_MIN_SIGMA <= sigma <= _MAX_SIGMA):
        return 1e12
    try:
        A_series = _invert_assets(equity_series, sigma, debt, rf, T, q)
    except CalibrationConvergenceError:
        return 1e12
    if np.any(A_series <= 0):
        return 1e12
    log_returns = np.diff(np.log(A_series))
    drift_increment = (mu - 0.5 * sigma * sigma) * dt
    var_increment = sigma * sigma * dt
    # Log density of N(drift_increment, var_increment) increments
    ll_returns = -0.5 * np.sum(
        np.log(2.0 * np.pi * var_increment) + (log_returns - drift_increment) ** 2 / var_increment
    )
    # Jacobian: d_E/d_A = Phi(d1)·e^(-qT_t); here T_t = T (constant maturity convention)
    sqrtT = float(np.sqrt(T))
    d1 = (np.log(A_series / debt) + (rf - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    jacobian_log = np.sum(np.log(np.maximum(A_series * norm_cdf(d1), 1e-300))) + (
        -q * T * A_series.size
    )
    ll = float(ll_returns - jacobian_log)
    if survivor_bias_correction:
        T_obs = (equity_series.size - 1) * dt
        log_psurv = log_survival_probability(
            asset_value_0=float(A_series[0]),
            debt=debt,
            drift=mu,
            sigma=sigma,
            T=T_obs,
        )
        ll = ll - log_psurv  # subtract log P(survive) ⇒ +(-log P) to neg LL
    return -ll


def _numeric_hessian(
    fn: callable,
    x: np.ndarray,
    *,
    eps: float = _HESSIAN_EPS,
) -> np.ndarray:
    """Central-difference Hessian of a scalar function at ``x``."""
    n = x.size
    H = np.zeros((n, n), dtype=np.float64)
    f0 = float(fn(x))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                xp = x.copy()
                xm = x.copy()
                xp[i] += eps
                xm[i] -= eps
                H[i, i] = (float(fn(xp)) - 2.0 * f0 + float(fn(xm))) / (eps * eps)
            else:
                xpp = x.copy()
                xpm = x.copy()
                xmp = x.copy()
                xmm = x.copy()
                xpp[i] += eps
                xpp[j] += eps
                xpm[i] += eps
                xpm[j] -= eps
                xmp[i] -= eps
                xmp[j] += eps
                xmm[i] -= eps
                xmm[j] -= eps
                val = (float(fn(xpp)) - float(fn(xpm)) - float(fn(xmp)) + float(fn(xmm))) / (
                    4.0 * eps * eps
                )
                H[i, j] = val
                H[j, i] = val
    return H


def duan_mle(
    *,
    equity_series: FloatArray,
    debt: float,
    rf: float,
    T: float,
    dt: float = 1.0 / ANNUALIZATION,
    dividend_yield: float = 0.0,
    survivor_bias_correction: bool = True,
    initial_sigma: float | None = None,
    initial_mu: float | None = None,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> CalibrationResult:
    """Run the Duan transformed-data MLE on an equity time series.

    Parameters
    ----------
    equity_series
        1-D array of equity values at uniform time steps.
    debt
        Default threshold (single liability at maturity ``T``).
    rf, T
        Risk-free rate and time-to-maturity (constant maturity convention).
    dt
        Sampling interval in years. Default 1/252 (daily).
    dividend_yield
        Continuous dividend yield ``q``.
    survivor_bias_correction
        If True, add ``-log P(survive)`` to the negative log-likelihood per
        Duan-Gauthier-Simonato-Zaanoun (2004). Default True.
    initial_sigma, initial_mu
        Optional warm-starts for the optimiser. If omitted, ``σ_A`` is
        initialised from the naive equity-vol proxy and ``μ`` from the
        sample mean of equity log-returns annualised.
    tol, max_iter
        Optimiser stopping criteria.
    """
    eq = np.asarray(equity_series, dtype=np.float64)
    if eq.ndim != 1 or eq.size < 30:
        raise InsufficientDataError(
            "Duan MLE needs an equity series of at least 30 observations",
            suggested_fix="Provide a longer history or use method='naive'.",
        )
    if debt <= 0 or T <= 0:
        raise MertonInputError("debt and T must be strictly positive")

    log_returns_eq = np.diff(np.log(eq))
    sigma0 = (
        float(initial_sigma)
        if initial_sigma is not None
        else float(log_returns_eq.std(ddof=1) * np.sqrt(1.0 / dt))
        * float(eq[-1])
        / (float(eq[-1]) + debt)
    )
    sigma0 = float(np.clip(sigma0, 1.5 * _MIN_SIGMA, _MAX_SIGMA - 0.1))
    mu0 = (
        float(initial_mu)
        if initial_mu is not None
        else float(log_returns_eq.mean() / dt) + 0.5 * sigma0 * sigma0
    )

    def neg_ll(params: np.ndarray) -> float:
        return _neg_log_likelihood(
            params,
            equity_series=eq,
            debt=debt,
            rf=rf,
            T=T,
            dt=dt,
            q=dividend_yield,
            survivor_bias_correction=survivor_bias_correction,
        )

    res = minimize(
        neg_ll,
        x0=np.array([mu0, sigma0]),
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0), (_MIN_SIGMA, _MAX_SIGMA)],
        options={"ftol": tol, "maxiter": max_iter},
    )
    if not res.success:
        raise CalibrationConvergenceError(
            f"Duan MLE optimiser failed: {res.message}",
            suggested_fix="Try a longer series, looser tol, or different warm-starts.",
        )

    mu_hat = float(res.x[0])
    sigma_hat = float(res.x[1])
    A_series = _invert_assets(eq, sigma_hat, debt, rf, T, dividend_yield)

    try:
        hess = _numeric_hessian(neg_ll, res.x)
        cov = np.linalg.inv(hess)
        if np.any(np.diag(cov) < 0):
            cov = None
    except (np.linalg.LinAlgError, ValueError):
        cov = None

    return CalibrationResult(
        asset_value=A_series,
        asset_vol=sigma_hat,
        asset_drift=mu_hat,
        log_likelihood=-float(res.fun),
        covariance=cov,
        n_iter=int(res.nit) if hasattr(res, "nit") else 0,
        converged=bool(res.success),
        method="duan_mle",
        diagnostics={
            "survivor_bias_correction": survivor_bias_correction,
            "n_obs": int(eq.size),
            "dt": float(dt),
            "param_order": ("asset_drift", "asset_vol"),
        },
    )


class DuanMLECalibrator(Calibrator):
    """OO wrapper around :func:`duan_mle`."""

    method = "duan_mle"

    def __init__(
        self,
        *,
        survivor_bias_correction: bool = True,
        tol: float = 1e-6,
        max_iter: int = 200,
    ) -> None:
        self.survivor_bias_correction = survivor_bias_correction
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, firm: Firm) -> CalibrationResult:
        equity = np.asarray(firm.equity, dtype=np.float64)
        if equity.ndim == 0 or equity.size < 30:
            raise InsufficientDataError(
                "Duan MLE requires Firm.equity to be a time series ≥ 30 obs.",
                suggested_fix=(
                    "Either supply equity as an array, or use method='vassalou_xing' "
                    "/ 'jmr_iterative' / 'naive' for snapshot data."
                ),
            )
        dp = firm.default_point_value()
        debt = float(dp.item() if np.ndim(dp) == 0 else float(np.mean(dp)))
        return duan_mle(
            equity_series=equity,
            debt=debt,
            rf=float(np.mean(np.asarray(firm.rf, dtype=np.float64))),
            T=float(firm.horizon),
            dividend_yield=float(np.mean(np.asarray(firm.dividend_yield, dtype=np.float64))),
            survivor_bias_correction=self.survivor_bias_correction,
            tol=self.tol,
            max_iter=self.max_iter,
        )


__all__ = ["DuanMLECalibrator", "duan_mle"]
