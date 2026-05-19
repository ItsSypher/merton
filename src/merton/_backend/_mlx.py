r"""MLX-backed kernels (Apple Silicon GPU via Metal).

Mirrors :mod:`merton._backend._numpy` semantics using
`mlx <https://ml-explore.github.io/mlx>`_. MLX is a NumPy-style array
framework with a unified-memory model: arrays live in memory the CPU
and GPU both see, so dispatching to it incurs no copies on Apple Silicon.

Normal CDF
----------
MLX has no ``scipy.stats``-style normal CDF, but it does have ``mx.erf``,
and ``Φ(x) = 0.5·(1 + erf(x / √2))`` is exact (modulo ``erf`` precision).
PDF is the standard ``exp(-x²/2)/√(2π)``.

Lazy evaluation
---------------
MLX is *lazy* by default — operations return a thunk; calling ``.item()``,
``.tolist()``, or ``mx.eval(...)`` is what materialises the answer. Our
kernels return arrays that the caller is expected to materialise when
they actually need the numeric value (e.g. via ``np.asarray(...)``).
"""

from __future__ import annotations

import mlx.core as mx

_SQRT_2 = float(2.0**0.5)
_INV_SQRT_2 = 1.0 / _SQRT_2
_INV_SQRT_2PI = 1.0 / float((2.0 * 3.141592653589793) ** 0.5)


def norm_cdf(x):  # type: ignore[no-untyped-def]
    """Standard normal CDF Φ(x) via ``mx.erf``."""
    return 0.5 * (1.0 + mx.erf(mx.array(x) * _INV_SQRT_2))


def norm_pdf(x):  # type: ignore[no-untyped-def]
    """Standard normal PDF φ(x)."""
    arr = mx.array(x)
    return _INV_SQRT_2PI * mx.exp(-0.5 * arr * arr)


def _d1_d2_core(A, sigma, D, r, T, q):  # type: ignore[no-untyped-def]
    sqrtT = mx.sqrt(T)
    d1 = (mx.log(A / D) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def d1_d2(A, sigma, D, r, T, q=0.0):  # type: ignore[no-untyped-def]
    """Return ``d1`` and ``d2`` from BSM (MLX version)."""
    A_arr = mx.array(A)
    sigma_arr = mx.array(sigma)
    D_arr = mx.array(D)
    r_arr = mx.array(r)
    T_arr = mx.array(T)
    q_arr = mx.array(q)
    return _d1_d2_core(A_arr, sigma_arr, D_arr, r_arr, T_arr, q_arr)


def equity_value(A, sigma, D, r, T, q=0.0):  # type: ignore[no-untyped-def]
    """Black-Scholes-Merton equity value via MLX."""
    d1, d2 = d1_d2(A, sigma, D, r, T, q)
    A_arr = mx.array(A)
    D_arr = mx.array(D)
    r_arr = mx.array(r)
    q_arr = mx.array(q)
    T_arr = mx.array(T)
    return A_arr * mx.exp(-q_arr * T_arr) * norm_cdf(d1) - D_arr * mx.exp(
        -r_arr * T_arr
    ) * norm_cdf(d2)


def distance_to_default_kernel(A, sigma, D, r, T, q=0.0):  # type: ignore[no-untyped-def]
    """``d_2`` of BSM = distance to default (MLX)."""
    _, d2 = d1_d2(A, sigma, D, r, T, q)
    return d2


def prob_of_default_kernel(dd):  # type: ignore[no-untyped-def]
    """Risk-neutral PD = Φ(-DD) on Apple Silicon."""
    return norm_cdf(-mx.array(dd))


__all__ = [
    "d1_d2",
    "distance_to_default_kernel",
    "equity_value",
    "norm_cdf",
    "norm_pdf",
    "prob_of_default_kernel",
]
