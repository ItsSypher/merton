"""Numba-JIT'd hot-path kernels.

Same maths as :mod:`merton._backend._numpy`, expressed in numba-friendly
form (``math.erf``-based normal CDF, no ``scipy`` calls). Compiled lazily on
first call; the wheel ships a pre-warmed Numba cache so users don't pay
JIT latency on import.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

INV_SQRT_2 = 1.0 / math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


@nb.njit(cache=True, fastmath=True, error_model="numpy", nogil=True, inline="always")
def _phi_scalar(x: float) -> float:
    """Standard normal CDF via ``math.erf``. Accurate to ~1e-15."""
    return 0.5 * (1.0 + math.erf(x * INV_SQRT_2))


@nb.njit(cache=True, fastmath=True, error_model="numpy", nogil=True, parallel=True)
def norm_cdf(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for i in nb.prange(x.size):  # type: ignore[attr-defined]
        out.flat[i] = _phi_scalar(x.flat[i])
    return out


@nb.njit(cache=True, fastmath=True, error_model="numpy", nogil=True, parallel=True)
def norm_pdf(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for i in nb.prange(x.size):  # type: ignore[attr-defined]
        v = x.flat[i]
        out.flat[i] = INV_SQRT_2PI * math.exp(-0.5 * v * v)
    return out


@nb.njit(cache=True, fastmath=True, error_model="numpy", nogil=True, parallel=True)
def _d1_d2_impl(
    A: np.ndarray,
    s: np.ndarray,
    D: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = A.size
    d1 = np.empty(n, dtype=np.float64)
    d2 = np.empty(n, dtype=np.float64)
    for i in nb.prange(n):  # type: ignore[attr-defined]
        sqrtT = math.sqrt(T.flat[i])
        s_i = s.flat[i]
        v = math.log(A.flat[i] / D.flat[i])
        d1_i = (v + (r.flat[i] - q.flat[i] + 0.5 * s_i * s_i) * T.flat[i]) / (s_i * sqrtT)
        d1[i] = d1_i
        d2[i] = d1_i - s_i * sqrtT
    return d1, d2


def _broadcast(arrays: tuple[np.ndarray | float, ...]) -> tuple[np.ndarray, ...]:
    """Broadcast inputs to a common shape and flatten for Numba."""
    bcast = np.broadcast_arrays(*[np.asarray(a, dtype=np.float64) for a in arrays])
    return tuple(np.ascontiguousarray(b).ravel() for b in bcast)


def d1_d2(
    asset_value: np.ndarray | float,
    asset_vol: np.ndarray | float,
    debt: np.ndarray | float,
    rf: np.ndarray | float,
    T: np.ndarray | float,
    dividend_yield: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    A_, s_, D_, r_, q_, T_ = _broadcast((asset_value, asset_vol, debt, rf, dividend_yield, T))
    shape = np.broadcast_shapes(
        np.shape(asset_value),
        np.shape(asset_vol),
        np.shape(debt),
        np.shape(rf),
        np.shape(dividend_yield),
        np.shape(T),
    )
    d1, d2 = _d1_d2_impl(A_, s_, D_, r_, q_, T_)
    if shape == ():
        return d1.reshape(())[()], d2.reshape(())[()]
    return d1.reshape(shape), d2.reshape(shape)


def equity_value(
    asset_value: np.ndarray | float,
    asset_vol: np.ndarray | float,
    debt: np.ndarray | float,
    rf: np.ndarray | float,
    T: np.ndarray | float,
    dividend_yield: np.ndarray | float = 0.0,
) -> np.ndarray:
    d1, d2 = d1_d2(asset_value, asset_vol, debt, rf, T, dividend_yield)
    A = np.asarray(asset_value, dtype=np.float64)
    D = np.asarray(debt, dtype=np.float64)
    r = np.asarray(rf, dtype=np.float64)
    q = np.asarray(dividend_yield, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    cdf_d1 = norm_cdf(np.atleast_1d(d1).astype(np.float64)).reshape(np.shape(d1))
    cdf_d2 = norm_cdf(np.atleast_1d(d2).astype(np.float64)).reshape(np.shape(d2))
    return A * np.exp(-q * T_) * cdf_d1 - D * np.exp(-r * T_) * cdf_d2


def distance_to_default_kernel(
    asset_value: np.ndarray | float,
    asset_vol: np.ndarray | float,
    debt: np.ndarray | float,
    rf: np.ndarray | float,
    T: np.ndarray | float,
    dividend_yield: np.ndarray | float = 0.0,
) -> np.ndarray:
    _, d2 = d1_d2(asset_value, asset_vol, debt, rf, T, dividend_yield)
    return d2


def prob_of_default_kernel(dd: np.ndarray | float) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(dd, dtype=np.float64))
    out = norm_cdf(-arr)
    if np.ndim(dd) == 0:
        return out.reshape(())[()]
    return out.reshape(np.shape(dd))


def warm_cache() -> None:
    """Pre-compile all JIT kernels with representative inputs.

    Called by the wheel build to materialise the Numba cache so end users
    don't pay the first-call compilation cost.
    """
    x = np.linspace(-3.0, 3.0, 64)
    norm_cdf(x)
    norm_pdf(x)
    A = np.full(64, 100.0)
    s = np.full(64, 0.25)
    D = np.full(64, 60.0)
    r = np.full(64, 0.04)
    q = np.zeros(64)
    T = np.ones(64)
    _d1_d2_impl(A, s, D, r, q, T)


__all__ = [
    "d1_d2",
    "distance_to_default_kernel",
    "equity_value",
    "norm_cdf",
    "norm_pdf",
    "prob_of_default_kernel",
    "warm_cache",
]
