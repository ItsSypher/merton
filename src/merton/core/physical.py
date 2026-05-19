r"""Risk-neutral ↔ physical probability-of-default conversion.

The Merton model under Black-Scholes assumptions produces *risk-neutral* PDs
(``Q``-measure). For credit-loss forecasting, banks typically need *physical*
PDs (``P``-measure). The two are related through the market price of risk:

.. math::

    DD_P = DD_Q + \lambda\, \sigma_A\, \sqrt{T}

where :math:`\lambda = (\mu - r) / \sigma_M` is the (asset-class) Sharpe
ratio. The physical PD is then :math:`\Phi(-DD_P)`.
"""

from __future__ import annotations

import numpy as np

from .._backend._numpy import norm_cdf, norm_ppf
from .._typing import ArrayLike, FloatArray
from ..exceptions import MertonInputError


def physical_pd(
    rn_pd: ArrayLike,
    asset_vol: ArrayLike,
    sharpe_ratio: ArrayLike,
    T: ArrayLike,
) -> FloatArray:
    r"""Convert a risk-neutral PD into a physical PD.

    Parameters
    ----------
    rn_pd
        Risk-neutral probability of default (decimal in [0, 1]).
    asset_vol
        Annualised asset volatility.
    sharpe_ratio
        Equity / asset-class Sharpe ratio ``λ = (μ - r) / σ``.
    T
        Horizon (years).
    """
    rn = np.asarray(rn_pd, dtype=np.float64)
    s = np.asarray(asset_vol, dtype=np.float64)
    lam = np.asarray(sharpe_ratio, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    if np.any(rn < 0) or np.any(rn > 1):
        raise MertonInputError("rn_pd must lie in [0, 1]")
    if np.any(T_ <= 0):
        raise MertonInputError("T must be strictly positive")
    # Clip to keep ppf finite at the tails.
    rn_clipped = np.clip(rn, 1e-15, 1.0 - 1e-15)
    dd_q = -norm_ppf(rn_clipped)
    dd_p = dd_q + lam * s * np.sqrt(T_)
    return norm_cdf(-dd_p)


__all__ = ["physical_pd"]
