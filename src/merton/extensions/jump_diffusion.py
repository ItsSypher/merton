r"""Merton-style jump-diffusion structural model (Zhou, 1997).

Adds a compound-Poisson jump component to the standard GBM asset
process. The closed-form Poisson-weighted series for the European-style
default probability follows from the same trick Merton (1976) used to
price options under jumps:

.. math::

    dV/V = (\mu - \lambda \kappa)\,dt + \sigma\,dW + (Y - 1)\,dN,

with ``N`` a Poisson process of intensity ``\lambda``, ``Y \sim \text{lognormal}(\mu_J, \sigma_J^2)``,
and ``\kappa = E[Y] - 1 = e^{\mu_J + \sigma_J^2/2} - 1`` chosen so the
total drift remains ``\mu``.

Conditional on exactly ``n`` jumps in ``[0, T]``, the asset log-return is
Gaussian with parameters

.. math::

    m_n = (\mu - \lambda \kappa - \sigma^2/2)\,T + n\,\mu_J, \qquad
    s_n^2 = \sigma^2 T + n\,\sigma_J^2.

The unconditional risk-neutral PD at horizon ``T`` is therefore

.. math::

    \mathrm{PD} = \sum_{n=0}^\infty
        \frac{e^{-\lambda T} (\lambda T)^n}{n!}
        \,\Phi\!\bigl(-d_2^{(n)}\bigr),

where ``d_2^{(n)}`` is the Merton-style distance to default using the
jump-adjusted drift and variance ``m_n``, ``s_n^2``.

References
----------
Merton, R. C. (1976). Option Pricing when Underlying Stock Returns are
Discontinuous. *Journal of Financial Economics* 3 (1-2), 125-144.

Zhou, C. (1997). A Jump-Diffusion Approach to Modeling Credit Risk and
Valuing Defaultable Securities. *Federal Reserve Working Paper* 97-15.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm

from .._typing import ArrayLike, FloatArray
from ..exceptions import MertonInputError
from .base import StructuralModel, StructuralResult

if TYPE_CHECKING:
    from ..core.firm import Firm


def jump_diffusion_pd(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
    *,
    jump_intensity: float = 0.5,
    jump_mean: float = -0.05,
    jump_std: float = 0.15,
    dividend_yield: ArrayLike = 0.0,
    max_terms: int = 50,
) -> FloatArray:
    r"""Merton-jump risk-neutral PD via the Poisson-weighted series.

    Parameters
    ----------
    asset_value, asset_vol, debt, rf, T
        Standard Merton inputs.
    jump_intensity
        ``\lambda`` — average number of jumps per year (default 0.5).
    jump_mean
        ``\mu_J`` — mean of the log-jump size (default -0.05 ≈ small
        negative shock).
    jump_std
        ``\sigma_J`` — std of the log-jump size (default 0.15).
    max_terms
        Truncation point for the infinite Poisson series. Values beyond
        the Poisson mean ``\lambda T + 6\sqrt{\lambda T}`` contribute
        negligible probability mass.
    """
    A = np.asarray(asset_value, dtype=np.float64)
    s = np.asarray(asset_vol, dtype=np.float64)
    D = np.asarray(debt, dtype=np.float64)
    r = np.asarray(rf, dtype=np.float64)
    q = np.asarray(dividend_yield, dtype=np.float64)
    T_arr = np.asarray(T, dtype=np.float64)
    if np.any(A <= 0) or np.any(s <= 0) or np.any(D <= 0) or np.any(T_arr <= 0):
        raise MertonInputError("asset_value, asset_vol, debt, T must be strictly positive")
    if jump_intensity < 0:
        raise MertonInputError("jump_intensity must be non-negative")
    if jump_std <= 0:
        raise MertonInputError("jump_std must be strictly positive")

    lam = float(jump_intensity)
    mu_J = float(jump_mean)
    sig_J = float(jump_std)
    kappa = math.exp(mu_J + 0.5 * sig_J * sig_J) - 1.0

    log_v = np.log(A / D)
    pd_total = np.zeros_like(
        np.broadcast_to(
            A, np.broadcast_shapes(A.shape, s.shape, D.shape, r.shape, q.shape, T_arr.shape)
        )
    )
    pd_total = np.asarray(pd_total, dtype=np.float64)
    lT = lam * T_arr
    # Truncate at max(20, λT + 6 √λT) — covers ≥ 1 − 1e-12 of the Poisson mass.
    n_truncate = int(
        np.maximum(max_terms, float(np.max(lT + 6 * np.sqrt(np.maximum(lT, 0.0)) + 1)))
    )
    n_truncate = min(n_truncate, 500)
    for n in range(n_truncate + 1):
        # Poisson weight log-pmf for numerical stability.
        log_w = -lT + n * np.log(np.maximum(lT, 1e-300)) - gammaln(n + 1)
        weight = np.exp(log_w)
        # Adjusted drift and variance conditional on n jumps.
        mean = (r - q - lam * kappa - 0.5 * s * s) * T_arr + n * mu_J
        var = s * s * T_arr + n * sig_J * sig_J
        std = np.sqrt(var)
        d2_n = (log_v + mean) / std
        pd_total = pd_total + weight * norm.cdf(-d2_n)
    return np.clip(pd_total, 0.0, 1.0)


def simulate_jump_diffusion(
    *,
    asset_value: float,
    drift: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    seed: int | None = None,
) -> np.ndarray:
    r"""Simulate Merton-jump asset paths. Returns shape ``(n_paths, n_steps + 1)``."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    kappa = math.exp(jump_mean + 0.5 * jump_std * jump_std) - 1.0
    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = asset_value
    for t in range(n_steps):
        z = rng.standard_normal(size=n_paths)
        n_jumps = rng.poisson(jump_intensity * dt, size=n_paths)
        jump_log = rng.normal(jump_mean, jump_std, size=n_paths)
        log_jump_contrib = np.where(n_jumps > 0, jump_log * n_jumps, 0.0)
        log_step = (
            (drift - jump_intensity * kappa - 0.5 * sigma * sigma) * dt
            + sigma * np.sqrt(dt) * z
            + log_jump_contrib
        )
        paths[:, t + 1] = paths[:, t] * np.exp(log_step)
    return paths


class JumpDiffusionModel(StructuralModel):
    """Zhou-style jump-diffusion structural model.

    Calibrates ``(A, σ_A)`` via JMR on the equity inputs (treating jumps
    as an extension of the Merton baseline), then layers the
    Poisson-weighted jump series on top for the PD.
    """

    method = "jump_diffusion"

    def __init__(
        self,
        *,
        jump_intensity: float = 0.5,
        jump_mean: float = -0.05,
        jump_std: float = 0.15,
        tol: float = 1e-8,
        max_iter: int = 200,
    ) -> None:
        self.jump_intensity = float(jump_intensity)
        self.jump_mean = float(jump_mean)
        self.jump_std = float(jump_std)
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, firm: Firm) -> StructuralResult:
        from ..calibration._solvers import solve_two_equation

        if firm.equity_vol is None:
            raise MertonInputError(
                "JumpDiffusionModel needs equity_vol on the Firm",
                suggested_fix="Pass equity_vol explicitly.",
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
            jump_diffusion_pd(
                a_val,
                sigma_a,
                debt,
                r,
                firm.horizon,
                jump_intensity=self.jump_intensity,
                jump_mean=self.jump_mean,
                jump_std=self.jump_std,
                dividend_yield=q,
            )
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
            method="jump_diffusion",
            diagnostics={
                "jump_intensity": self.jump_intensity,
                "jump_mean": self.jump_mean,
                "jump_std": self.jump_std,
            },
        )


__all__ = ["JumpDiffusionModel", "jump_diffusion_pd", "simulate_jump_diffusion"]
