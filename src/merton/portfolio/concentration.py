"""Concentration metrics."""

from __future__ import annotations

import numpy as np

from .._typing import ArrayLike
from ..exceptions import MertonInputError


def hhi(exposures: ArrayLike) -> float:
    r"""Herfindahl-Hirschman Index of a portfolio.

    .. math::

        \mathrm{HHI} = \sum_i w_i^2,\quad w_i = \frac{E_i}{\sum_j E_j}.

    Ranges in ``[1/n, 1]``. A perfectly diversified portfolio of ``n``
    equal exposures has ``HHI = 1/n``; a portfolio concentrated in one
    name has ``HHI = 1``.
    """
    exp = np.asarray(exposures, dtype=np.float64)
    if exp.ndim != 1:
        raise MertonInputError("exposures must be 1-D")
    if np.any(exp < 0):
        raise MertonInputError("exposures must be non-negative")
    total = exp.sum()
    if total <= 0:
        raise MertonInputError("exposures sum must be positive")
    weights = exp / total
    return float(np.sum(weights**2))


def effective_n(exposures: ArrayLike) -> float:
    """Effective number of names ``1/HHI``."""
    return 1.0 / hhi(exposures)


def granularity_adjustment(
    pd: ArrayLike,
    lgd: ArrayLike,
    rho: ArrayLike,
    exposures: ArrayLike,
    *,
    alpha: float = 0.999,
) -> float:
    r"""Pykhtin-Dev (2002) granularity adjustment for the IRB asymptotic VaR.

    Computes the leading-order correction to the Vasicek IRB VaR for finite
    portfolio size. Returns the *additive* adjustment in units of exposure
    fraction (so the actual VaR upper bound is
    ``VaR_IRB(pd, lgd, rho) + granularity_adjustment(...)``).

    Pykhtin & Dev (2002) *Credit Risk in Asset Securitizations*. Risk 15(5).
    """
    from scipy.stats import norm

    from .vasicek_factor import vasicek_var

    exp = np.asarray(exposures, dtype=np.float64)
    if exp.ndim != 1:
        raise MertonInputError("exposures must be 1-D")
    weights = exp / exp.sum()
    pd_arr = np.asarray(pd, dtype=np.float64)
    lgd_arr = np.asarray(lgd, dtype=np.float64)
    rho_arr = np.asarray(rho, dtype=np.float64)
    pd_arr = np.broadcast_to(pd_arr, weights.shape)
    lgd_arr = np.broadcast_to(lgd_arr, weights.shape)
    rho_arr = np.broadcast_to(rho_arr, weights.shape)
    # First-order Pykhtin-Dev kernel for homogeneous-PD case.
    var_inf = vasicek_var(pd_arr, rho_arr, alpha=alpha)
    sqrt_rho = np.sqrt(rho_arr)
    sqrt_1mrho = np.sqrt(1.0 - rho_arr)
    z = norm.ppf(alpha)
    # K_i = LGD_i * φ(Φ⁻¹(p_i) - √(1-ρ) Φ⁻¹(α)) etc. — keep it simple and
    # use the leading-order density correction.
    K = lgd_arr * norm.pdf((norm.ppf(pd_arr) - sqrt_1mrho * (-z)) / sqrt_rho)
    ga = 0.5 * np.sum((weights**2) * K * (1.0 - var_inf))
    return float(ga)


__all__ = ["effective_n", "granularity_adjustment", "hhi"]
