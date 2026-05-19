"""Black-Scholes-Merton pricing helpers (public, backend-dispatched)."""

from __future__ import annotations

import numpy as np

from .._backend import get_kernel, resolve
from .._typing import ArrayLike, FloatArray


def _size(*arrays: object) -> int:
    return int(
        np.broadcast_shapes(*(np.shape(a) for a in arrays)).count(0)
        or max((np.size(np.asarray(a)) for a in arrays), default=1)
    )


def equity_value(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
    *,
    dividend_yield: ArrayLike = 0.0,
    backend: str | None = None,
) -> FloatArray:
    """Black-Scholes-Merton equity value treating equity as a call on assets.

    Parameters
    ----------
    asset_value
        Firm asset value ``A`` (currency units).
    asset_vol
        Annualised asset volatility ``σ_A`` (decimal, e.g. ``0.25``).
    debt
        Face value of debt at maturity ``D`` (the default threshold).
    rf
        Risk-free rate ``r`` (annualised, continuously-compounded decimal).
    T
        Time to debt maturity (years).
    dividend_yield
        Continuous dividend yield ``q`` (decimal). Defaults to 0.
    backend
        Override the backend dispatch. Default: auto.

    Returns
    -------
    FloatArray
        ``E = A·e^(-qT)·Φ(d1) - D·e^(-rT)·Φ(d2)``.
    """
    chosen = resolve(
        asset_value,
        asset_vol,
        debt,
        rf,
        T,
        backend=backend,
        size=_size(asset_value, asset_vol, debt, rf, T),
    )
    kernel = get_kernel(chosen, "equity_value")
    return kernel(asset_value, asset_vol, debt, rf, T, dividend_yield)


def d1_d2(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
    *,
    dividend_yield: ArrayLike = 0.0,
    backend: str | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return BSM's ``d1`` and ``d2``."""
    chosen = resolve(
        asset_value,
        asset_vol,
        debt,
        rf,
        T,
        backend=backend,
        size=_size(asset_value, asset_vol, debt, rf, T),
    )
    kernel = get_kernel(chosen, "d1_d2")
    return kernel(asset_value, asset_vol, debt, rf, T, dividend_yield)


__all__ = ["d1_d2", "equity_value"]
