"""Risk-neutral PD sensitivities (closed form).

Each sensitivity is the partial derivative of ``PD = Φ(-d₂)`` with respect
to one input, evaluated at the supplied operating point.
"""

from __future__ import annotations

import numpy as np

from .._backend._numpy import d1_d2, norm_pdf
from .._typing import ArrayLike, FloatArray


def pd_leverage_sensitivity(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
    *,
    dividend_yield: ArrayLike = 0.0,
) -> FloatArray:
    """∂PD/∂D = φ(-d₂) · 1 / (D · σ_A · √T).

    Higher debt ⇒ higher PD. The sign is positive.
    """
    _, d2 = d1_d2(asset_value, asset_vol, debt, rf, T, dividend_yield)
    D = np.asarray(debt, dtype=np.float64)
    s = np.asarray(asset_vol, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    return norm_pdf(-d2) / (D * s * np.sqrt(T_))


def pd_vol_sensitivity(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
    *,
    dividend_yield: ArrayLike = 0.0,
) -> FloatArray:
    """∂PD/∂σ_A in closed form."""
    d1, d2 = d1_d2(asset_value, asset_vol, debt, rf, T, dividend_yield)
    s = np.asarray(asset_vol, dtype=np.float64)
    # ∂d2/∂σ = -d1/σ ; ∂PD/∂σ = -φ(-d2) · ∂d2/∂σ , PD = Φ(-d2)
    dd2_dsigma = -d1 / s
    return -norm_pdf(-d2) * dd2_dsigma


def pd_rate_sensitivity(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
    *,
    dividend_yield: ArrayLike = 0.0,
) -> FloatArray:
    """∂PD/∂r — typically negative: higher rates push the drift up, PD down."""
    _, d2 = d1_d2(asset_value, asset_vol, debt, rf, T, dividend_yield)
    s = np.asarray(asset_vol, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    sqrtT = np.sqrt(T_)
    dd2_dr = sqrtT / s
    return -norm_pdf(-d2) * dd2_dr


__all__ = [
    "pd_leverage_sensitivity",
    "pd_rate_sensitivity",
    "pd_vol_sensitivity",
]
