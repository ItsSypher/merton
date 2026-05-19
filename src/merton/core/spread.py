"""Implied credit spread from a Merton-model PD."""

from __future__ import annotations

import numpy as np

from .._typing import ArrayLike, FloatArray
from ..exceptions import MertonInputError


def implied_credit_spread(
    pd: ArrayLike,
    T: ArrayLike,
    lgd: ArrayLike = 0.6,
    *,
    in_bps: bool = True,
) -> FloatArray:
    r"""Continuous-compounding credit spread implied by a horizon PD.

    .. math::

        s = -\frac{1}{T} \ln\!\bigl(1 - \text{PD} \cdot \text{LGD}\bigr)

    Parameters
    ----------
    pd
        Cumulative probability of default over the horizon (decimal).
    T
        Horizon in years (must be > 0).
    lgd
        Loss given default (decimal, default 0.6 ≡ 60 % loss / 40 % recovery).
    in_bps
        Return basis points (``True``, default) instead of decimal.

    Returns
    -------
    FloatArray
        Annualised credit spread.

    Examples
    --------
    >>> float(round(implied_credit_spread(0.01, 1.0, lgd=0.6), 2))
    60.18
    """
    pd_arr = np.asarray(pd, dtype=np.float64)
    T_arr = np.asarray(T, dtype=np.float64)
    lgd_arr = np.asarray(lgd, dtype=np.float64)
    if np.any(T_arr <= 0):
        raise MertonInputError("T must be strictly positive")
    if np.any(lgd_arr < 0) or np.any(lgd_arr > 1):
        raise MertonInputError("lgd must lie in [0, 1]")
    if np.any(pd_arr < 0) or np.any(pd_arr > 1):
        raise MertonInputError("pd must lie in [0, 1]")
    product = np.clip(pd_arr * lgd_arr, 0.0, 1.0 - 1e-16)
    spread = -np.log1p(-product) / T_arr
    return spread * 10_000.0 if in_bps else spread


__all__ = ["implied_credit_spread"]
