"""CuPy-backed kernels (NVIDIA GPU via CUDA).

Mirrors :mod:`merton._backend._numpy` semantics using CuPy as a drop-in
NumPy replacement. The normal CDF uses ``cupyx.scipy.special.ndtr`` — the
same kernel SciPy uses on CPU, just dispatched to the device.

The module is **lazy-imported**: the package never imports CuPy at top
level. We only get here when:

- ``merton[gpu]`` is installed and a CUDA device is visible.
- ``MERTON_BACKEND=cupy`` is set (or the user explicitly passes
  ``backend="cupy"``), or
- the inputs are already :class:`cupy.ndarray` instances.

Performance notes
-----------------
CuPy kernels are *very* fast for large arrays (10⁵+ elements) but pay a
non-trivial Python-side launch latency per call. The dispatch layer in
:func:`merton._backend.resolve` only routes to CuPy when the input shape
or the user's explicit ``backend="cupy"`` warrants it.
"""

from __future__ import annotations

import cupy as cp
from cupyx.scipy.special import ndtr as _ndtr


def norm_cdf(x):  # type: ignore[no-untyped-def]
    """Standard normal CDF Φ(x) on the GPU."""
    return _ndtr(x)


def norm_pdf(x):  # type: ignore[no-untyped-def]
    """Standard normal PDF φ(x) on the GPU."""
    arr = cp.asarray(x, dtype=cp.float64)
    return cp.exp(-0.5 * arr * arr) / cp.sqrt(2.0 * cp.pi)


def _d1_d2_core(A, sigma, D, r, T, q):  # type: ignore[no-untyped-def]
    sqrtT = cp.sqrt(T)
    d1 = (cp.log(A / D) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def d1_d2(A, sigma, D, r, T, q=0.0):  # type: ignore[no-untyped-def]
    """Return ``d1`` and ``d2`` from BSM (CuPy version)."""
    A_arr = cp.asarray(A, dtype=cp.float64)
    sigma_arr = cp.asarray(sigma, dtype=cp.float64)
    D_arr = cp.asarray(D, dtype=cp.float64)
    r_arr = cp.asarray(r, dtype=cp.float64)
    T_arr = cp.asarray(T, dtype=cp.float64)
    q_arr = cp.asarray(q, dtype=cp.float64)
    return _d1_d2_core(A_arr, sigma_arr, D_arr, r_arr, T_arr, q_arr)


def equity_value(A, sigma, D, r, T, q=0.0):  # type: ignore[no-untyped-def]
    """Black-Scholes-Merton equity value on the GPU."""
    d1, d2 = d1_d2(A, sigma, D, r, T, q)
    A_arr = cp.asarray(A, dtype=cp.float64)
    D_arr = cp.asarray(D, dtype=cp.float64)
    r_arr = cp.asarray(r, dtype=cp.float64)
    q_arr = cp.asarray(q, dtype=cp.float64)
    T_arr = cp.asarray(T, dtype=cp.float64)
    return A_arr * cp.exp(-q_arr * T_arr) * _ndtr(d1) - D_arr * cp.exp(-r_arr * T_arr) * _ndtr(d2)


def distance_to_default_kernel(A, sigma, D, r, T, q=0.0):  # type: ignore[no-untyped-def]
    """``d_2`` of BSM = distance to default (GPU)."""
    _, d2 = d1_d2(A, sigma, D, r, T, q)
    return d2


def prob_of_default_kernel(dd):  # type: ignore[no-untyped-def]
    """Risk-neutral PD = Φ(-DD) on the GPU."""
    return _ndtr(-cp.asarray(dd, dtype=cp.float64))


__all__ = [
    "d1_d2",
    "distance_to_default_kernel",
    "equity_value",
    "norm_cdf",
    "norm_pdf",
    "prob_of_default_kernel",
]
