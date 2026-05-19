"""JAX-backed kernels.

Mirrors :mod:`merton._backend._numpy` semantics but uses ``jax.numpy``.
All public functions are wrapped with :func:`jax.jit` so the first call
compiles a fused XLA program for the given input shapes. Subsequent calls
re-use the cached compilation.

The module is imported **lazily** — the package never imports JAX at top
level. We only get here when:

- ``merton.[jax]`` is installed,
- ``MERTON_BACKEND=jax`` is set (or the user explicitly passes
  ``backend="jax"``), or
- the inputs are JAX arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as _jnorm


@jax.jit
def norm_cdf(x):  # type: ignore[no-untyped-def]
    """Standard normal CDF, JIT-compiled."""
    return _jnorm.cdf(x)


@jax.jit
def norm_pdf(x):  # type: ignore[no-untyped-def]
    """Standard normal PDF, JIT-compiled."""
    return _jnorm.pdf(x)


def _d1_d2_core(A, sigma, D, r, T, q):  # type: ignore[no-untyped-def]
    sqrtT = jnp.sqrt(T)
    d1 = (jnp.log(A / D) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


@jax.jit
def d1_d2(A, sigma, D, r, T, q=0.0):  # type: ignore[no-untyped-def]
    """Return ``d1`` and ``d2`` from BSM."""
    return _d1_d2_core(
        jnp.asarray(A),
        jnp.asarray(sigma),
        jnp.asarray(D),
        jnp.asarray(r),
        jnp.asarray(T),
        jnp.asarray(q),
    )


@jax.jit
def equity_value(A, sigma, D, r, T, q=0.0):  # type: ignore[no-untyped-def]
    """``E = A·e^(-qT)·Φ(d1) - D·e^(-rT)·Φ(d2)``."""
    d1, d2 = _d1_d2_core(
        jnp.asarray(A),
        jnp.asarray(sigma),
        jnp.asarray(D),
        jnp.asarray(r),
        jnp.asarray(T),
        jnp.asarray(q),
    )
    return jnp.asarray(A) * jnp.exp(-jnp.asarray(q) * jnp.asarray(T)) * _jnorm.cdf(
        d1
    ) - jnp.asarray(D) * jnp.exp(-jnp.asarray(r) * jnp.asarray(T)) * _jnorm.cdf(d2)


@jax.jit
def distance_to_default_kernel(A, sigma, D, r, T, q=0.0):  # type: ignore[no-untyped-def]
    _, d2 = _d1_d2_core(
        jnp.asarray(A),
        jnp.asarray(sigma),
        jnp.asarray(D),
        jnp.asarray(r),
        jnp.asarray(T),
        jnp.asarray(q),
    )
    return d2


@jax.jit
def prob_of_default_kernel(dd):  # type: ignore[no-untyped-def]
    return _jnorm.cdf(-jnp.asarray(dd))


__all__ = [
    "d1_d2",
    "distance_to_default_kernel",
    "equity_value",
    "norm_cdf",
    "norm_pdf",
    "prob_of_default_kernel",
]
