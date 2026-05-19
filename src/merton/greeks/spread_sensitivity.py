"""Sensitivity of the implied credit spread to its inputs.

The spread function is ``s = -ln(1 - PD·LGD) / T``. Composing with the chain
rule yields ∂s/∂x for x ∈ {asset_value, asset_vol, debt, rf, T} as a
PD-times-Jacobian-scalar formula.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .._typing import ArrayLike, FloatArray
from ..core.distance import distance_to_default, prob_of_default
from ..exceptions import MertonInputError
from .pd_sensitivity import (
    pd_leverage_sensitivity,
    pd_rate_sensitivity,
    pd_vol_sensitivity,
)

SpreadInput = Literal["leverage", "vol", "rate"]


def spread_sensitivity(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
    *,
    wrt: SpreadInput,
    lgd: ArrayLike = 0.6,
    dividend_yield: ArrayLike = 0.0,
) -> FloatArray:
    """∂(spread)/∂(input) where input is one of leverage, vol, rate."""
    if wrt not in ("leverage", "vol", "rate"):
        raise MertonInputError(
            f"Unknown sensitivity wrt={wrt!r}",
            suggested_fix="Pass one of 'leverage', 'vol', 'rate'.",
        )
    dd = distance_to_default(asset_value, asset_vol, debt, rf, T, dividend_yield=dividend_yield)
    pd = prob_of_default(dd)
    L = np.asarray(lgd, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    factor = L / (T_ * (1.0 - pd * L))
    if wrt == "leverage":
        dpd = pd_leverage_sensitivity(
            asset_value, asset_vol, debt, rf, T, dividend_yield=dividend_yield
        )
    elif wrt == "vol":
        dpd = pd_vol_sensitivity(asset_value, asset_vol, debt, rf, T, dividend_yield=dividend_yield)
    else:  # "rate"
        dpd = pd_rate_sensitivity(
            asset_value, asset_vol, debt, rf, T, dividend_yield=dividend_yield
        )
    return factor * dpd  # in decimal; multiply by 10000 for bps


__all__ = ["SpreadInput", "spread_sensitivity"]
