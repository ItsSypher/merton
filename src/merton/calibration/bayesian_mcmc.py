r"""Bayesian MCMC calibration via :mod:`emcee`.

Where :func:`merton.calibration.duan_mle` returns a single point estimate
plus asymptotic standard errors, the Bayesian MCMC calibrator returns a
**posterior distribution** over ``(\mu, \sigma_A)`` — which propagates
cleanly into credible intervals on DD, PD, and credit spread.

The likelihood is the Duan (1994) transformed-data likelihood; the
default priors are uniform on bounded ranges (``\mu \in [-5, 5]``,
``\sigma_A \in [0.001, 5]``). Pass custom priors via the ``prior_log_pdf``
argument.

Implementation
--------------
We use :class:`emcee.EnsembleSampler` with the affine-invariant stretch
move. The walker count defaults to ``2 * ndim`` (the emcee minimum) but
32+ is recommended for tight posteriors.

References
----------
Goodman, J., Weare, J. (2010). Ensemble samplers with affine invariance.
*Communications in Applied Mathematics and Computational Science* 5 (1).

Foreman-Mackey, D., Hogg, D. W., Lang, D., Goodman, J. (2013).
emcee: The MCMC Hammer. *PASP* 125 (925), 306-312.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .._typing import FloatArray
from ..exceptions import InsufficientDataError, MertonInputError
from .base import CalibrationResult, Calibrator
from .covariance import ConfInt

if TYPE_CHECKING:
    from ..core.firm import Firm


# Default prior bounds — wide enough to encompass essentially any plausible
# firm. Users with strong priors can pass a custom log-pdf callable.
DEFAULT_MU_RANGE = (-5.0, 5.0)
DEFAULT_SIGMA_RANGE = (1e-3, 5.0)


@dataclass(slots=True, frozen=True)
class BayesianCalibrationResult(CalibrationResult):
    """Output of :func:`bayesian_mcmc`. Extends ``CalibrationResult`` with
    the posterior chain plus credible-interval helpers."""

    chain: FloatArray = field(default_factory=lambda: np.empty((0, 0)))
    log_probs: FloatArray = field(default_factory=lambda: np.empty(0))
    acceptance_fraction: float = 0.0
    autocorr_time: FloatArray | None = None

    def credible_interval(
        self,
        param: str,
        level: float = 0.95,
    ) -> ConfInt:
        """Empirical highest-density / percentile credible interval."""
        if not 0 < level < 1:
            raise MertonInputError("level must lie in (0, 1)")
        if param == "asset_drift":
            samples = self.chain[:, 0]
        elif param == "asset_vol":
            samples = self.chain[:, 1]
        else:
            raise MertonInputError(
                f"unknown param {param!r}",
                suggested_fix="choose 'asset_drift' or 'asset_vol'.",
            )
        alpha = (1.0 - level) / 2.0
        lo = float(np.quantile(samples, alpha))
        hi = float(np.quantile(samples, 1.0 - alpha))
        return ConfInt(
            lower=lo,
            upper=hi,
            level=level,
            method="mcmc",
            se=float(np.std(samples, ddof=1)),
        )


def _default_log_prior(
    theta: np.ndarray,
    mu_range: tuple[float, float],
    sigma_range: tuple[float, float],
) -> float:
    mu, sigma = float(theta[0]), float(theta[1])
    if not (mu_range[0] <= mu <= mu_range[1]):
        return -np.inf
    if not (sigma_range[0] <= sigma <= sigma_range[1]):
        return -np.inf
    return 0.0  # uniform on the box


def bayesian_mcmc(
    *,
    equity_series: FloatArray,
    debt: float,
    rf: float,
    T: float,
    dt: float = 1.0 / 252.0,
    dividend_yield: float = 0.0,
    survivor_bias_correction: bool = False,
    n_walkers: int = 32,
    n_steps: int = 2000,
    burn_in: int = 500,
    thin: int = 5,
    seed: int | None = None,
    mu_range: tuple[float, float] = DEFAULT_MU_RANGE,
    sigma_range: tuple[float, float] = DEFAULT_SIGMA_RANGE,
    prior_log_pdf: Callable[[np.ndarray], float] | None = None,
) -> BayesianCalibrationResult:
    r"""Sample the Bayesian posterior of ``(μ, σ_A)`` via emcee.

    Parameters
    ----------
    equity_series
        1-D array of equity observations (≥ 30 points recommended).
    debt, rf, T, dt, dividend_yield
        Standard Duan-MLE inputs.
    survivor_bias_correction
        Pass to the underlying likelihood. Defaults to ``False`` because
        the correction makes the optimiser landscape harder to sample.
    n_walkers, n_steps, burn_in, thin, seed
        emcee parameters; sane defaults from the user guide.
    mu_range, sigma_range
        Bounds for the default uniform prior. Ignored when
        ``prior_log_pdf`` is passed.
    prior_log_pdf
        Optional custom log-prior. Must accept a 1-D array
        ``[μ, σ_A]`` and return a float.
    """
    try:
        import emcee
    except ImportError as err:  # pragma: no cover - optional extra
        raise ImportError(
            'merton.calibration.bayesian_mcmc requires merton[mcmc]: pip install "merton[mcmc]"'
        ) from err
    from .duan_mle import _neg_log_likelihood

    eq = np.asarray(equity_series, dtype=np.float64)
    if eq.ndim != 1 or eq.size < 30:
        raise InsufficientDataError(
            "Bayesian MCMC needs an equity series of at least 30 observations",
        )
    if debt <= 0 or T <= 0:
        raise MertonInputError("debt and T must be strictly positive")
    if n_walkers < 4:
        raise MertonInputError("n_walkers must be ≥ 4 (emcee minimum)")
    if n_steps <= burn_in:
        raise MertonInputError("n_steps must exceed burn_in")

    prior_fn = prior_log_pdf or (lambda theta: _default_log_prior(theta, mu_range, sigma_range))

    def log_prob(theta: np.ndarray) -> float:
        lp = prior_fn(theta)
        if not np.isfinite(lp):
            return -np.inf
        nll = _neg_log_likelihood(
            theta,
            equity_series=eq,
            debt=debt,
            rf=rf,
            T=T,
            dt=dt,
            q=dividend_yield,
            survivor_bias_correction=survivor_bias_correction,
        )
        if not np.isfinite(nll) or nll >= 1e11:
            return -np.inf
        return lp + (-nll)

    rng = np.random.default_rng(seed)
    # Initialise walkers in a tight ball around a naive guess.
    log_returns = np.diff(np.log(eq))
    sigma_init = float(log_returns.std(ddof=1) * np.sqrt(1.0 / dt) * eq[-1] / (eq[-1] + debt))
    sigma_init = float(np.clip(sigma_init, sigma_range[0] + 0.01, sigma_range[1] - 0.01))
    mu_init = float(
        np.clip(
            log_returns.mean() / dt + 0.5 * sigma_init**2, mu_range[0] + 0.01, mu_range[1] - 0.01
        )
    )
    p0 = rng.normal(loc=[mu_init, sigma_init], scale=[0.05, 0.02], size=(n_walkers, 2))
    # Clip to priors so no walker starts outside.
    p0[:, 0] = np.clip(p0[:, 0], mu_range[0] + 1e-3, mu_range[1] - 1e-3)
    p0[:, 1] = np.clip(p0[:, 1], sigma_range[0] + 1e-3, sigma_range[1] - 1e-3)

    sampler = emcee.EnsembleSampler(n_walkers, 2, log_prob)
    sampler.run_mcmc(p0, n_steps, progress=False)

    chain = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, thin=thin, flat=True)
    try:
        tau = sampler.get_autocorr_time(quiet=True)
    except Exception:
        tau = None

    mu_post_mean = float(chain[:, 0].mean())
    sigma_post_mean = float(chain[:, 1].mean())
    # MAP estimate.
    best = int(np.argmax(log_probs))
    mu_map = float(chain[best, 0])
    sigma_map = float(chain[best, 1])
    asset_drift = mu_map

    # Recover the asset value path at the MAP estimate.
    from .duan_mle import _invert_assets

    a_series = _invert_assets(eq, sigma_map, debt, rf, T, dividend_yield)

    return BayesianCalibrationResult(
        asset_value=a_series,
        asset_vol=sigma_map,
        asset_drift=asset_drift,
        log_likelihood=float(log_probs.max()),
        covariance=np.cov(chain.T),
        n_iter=int(n_steps),
        converged=True,
        method="bayesian_mcmc",
        diagnostics={
            "posterior_mean_mu": mu_post_mean,
            "posterior_mean_sigma": sigma_post_mean,
            "map_mu": mu_map,
            "map_sigma": sigma_map,
            "n_walkers": n_walkers,
            "burn_in": burn_in,
            "thin": thin,
        },
        chain=chain,
        log_probs=log_probs,
        acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
        autocorr_time=tau,
    )


class BayesianMCMCCalibrator(Calibrator):
    """OO wrapper around :func:`bayesian_mcmc`."""

    method = "bayesian_mcmc"

    def __init__(
        self,
        *,
        n_walkers: int = 32,
        n_steps: int = 2000,
        burn_in: int = 500,
        thin: int = 5,
        seed: int | None = None,
        tol: float = 1e-6,
        max_iter: int = 200,
    ) -> None:
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.burn_in = burn_in
        self.thin = thin
        self.seed = seed
        self.tol = tol  # ignored
        self.max_iter = max_iter  # ignored

    def fit(self, firm: Firm) -> BayesianCalibrationResult:
        equity = np.asarray(firm.equity, dtype=np.float64)
        if equity.ndim == 0 or equity.size < 30:
            raise InsufficientDataError(
                "Bayesian MCMC requires Firm.equity to be a series ≥ 30 obs.",
            )
        dp_arr = firm.default_point_value()
        debt = float(dp_arr.item() if np.ndim(dp_arr) == 0 else float(np.mean(dp_arr)))
        return bayesian_mcmc(
            equity_series=equity,
            debt=debt,
            rf=float(np.mean(np.asarray(firm.rf, dtype=np.float64))),
            T=float(firm.horizon),
            dividend_yield=float(np.mean(np.asarray(firm.dividend_yield, dtype=np.float64))),
            n_walkers=self.n_walkers,
            n_steps=self.n_steps,
            burn_in=self.burn_in,
            thin=self.thin,
            seed=self.seed,
        )


__all__ = [
    "DEFAULT_MU_RANGE",
    "DEFAULT_SIGMA_RANGE",
    "BayesianCalibrationResult",
    "BayesianMCMCCalibrator",
    "bayesian_mcmc",
]
