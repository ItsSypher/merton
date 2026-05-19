"""Autodiff Greeks via JAX.

Each Greek is computed by composing :func:`jax.grad` with the BSM equity
or PD expression, then :func:`jax.vmap` to vectorise across input arrays
and :func:`jax.jit` to compile the result. This is useful for:

- Validating the closed-form Greeks in :mod:`merton.greeks.equity` and
  :mod:`merton.greeks.pd_sensitivity` (they should agree to ~1e-7).
- Differentiating *novel* BSM extensions where deriving Greeks by hand is
  error-prone.

The module is only importable when ``merton[jax]`` is installed; importing
it without JAX raises a clean ``ImportError``.

Examples
--------
>>> from merton.greeks import autodiff  # doctest: +SKIP
>>> import numpy as np  # doctest: +SKIP
>>> A = np.array([100.0, 150.0, 80.0])  # doctest: +SKIP
>>> autodiff.equity_delta_ad(A, 0.25, 60.0, 0.04, 1.0)  # doctest: +SKIP
"""

from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.stats import norm as _jnorm
except ImportError as _err:  # pragma: no cover - optional dep
    raise ImportError(
        'merton.greeks.autodiff requires the JAX extra: `pip install "merton[jax]"`.'
    ) from _err

# JAX defaults to float32; structural-credit Greeks need float64 to match
# the closed-form implementations to single-precision tolerance. Enable
# x64 once at import time. (Setting it later via jax.config.update would
# warn that some jit caches are stale.)
jax.config.update("jax_enable_x64", True)


def _equity_value(A, sigma, D, r, T, q):  # type: ignore[no-untyped-def]
    sqrtT = jnp.sqrt(T)
    d1 = (jnp.log(A / D) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return A * jnp.exp(-q * T) * _jnorm.cdf(d1) - D * jnp.exp(-r * T) * _jnorm.cdf(d2)


def _dd(A, sigma, D, r, T, q):  # type: ignore[no-untyped-def]
    sqrtT = jnp.sqrt(T)
    return (jnp.log(A / D) + (r - q - 0.5 * sigma * sigma) * T) / (sigma * sqrtT)


def _pd(A, sigma, D, r, T, q):  # type: ignore[no-untyped-def]
    return _jnorm.cdf(-_dd(A, sigma, D, r, T, q))


def _grad_then_vmap_then_jit(fn, *, argnums: int, in_axes):  # type: ignore[no-untyped-def]
    return jax.jit(jax.vmap(jax.grad(fn, argnums=argnums), in_axes=in_axes))


# Greek table: (name, function, argnum_of_x_to_diff_against, default in_axes)
# in_axes carries 0 for asset_value/asset_vol/debt (broadcastable arrays) and
# None for rf/T/q (scalars).
_VMAP_AXES_FULL = (0, 0, 0, None, None, None)
_VMAP_AXES_AVOL = (0, 0, 0, None, None, None)


_equity_delta_fn = _grad_then_vmap_then_jit(_equity_value, argnums=0, in_axes=_VMAP_AXES_FULL)
_equity_vega_fn = _grad_then_vmap_then_jit(_equity_value, argnums=1, in_axes=_VMAP_AXES_FULL)
_equity_rho_fn = _grad_then_vmap_then_jit(_equity_value, argnums=3, in_axes=_VMAP_AXES_FULL)
_equity_theta_fn = _grad_then_vmap_then_jit(_equity_value, argnums=4, in_axes=_VMAP_AXES_FULL)

# Gamma = ∂²E/∂A² = ∂/∂A (∂E/∂A); compose grad twice.
_equity_gamma_fn = jax.jit(
    jax.vmap(jax.grad(jax.grad(_equity_value, argnums=0), argnums=0), in_axes=_VMAP_AXES_FULL)
)

_pd_dleverage_fn = _grad_then_vmap_then_jit(_pd, argnums=2, in_axes=_VMAP_AXES_FULL)
_pd_dvol_fn = _grad_then_vmap_then_jit(_pd, argnums=1, in_axes=_VMAP_AXES_FULL)
_pd_drate_fn = _grad_then_vmap_then_jit(_pd, argnums=3, in_axes=_VMAP_AXES_FULL)


def _broadcast_inputs(asset_value, asset_vol, debt, rf, T, dividend_yield):  # type: ignore[no-untyped-def]
    """Promote inputs to 1-D JAX arrays of the same length."""
    arrays = [
        jnp.atleast_1d(jnp.asarray(a, dtype=jnp.float64)) for a in (asset_value, asset_vol, debt)
    ]
    n = max(a.shape[0] for a in arrays)
    arrays = [jnp.broadcast_to(a, (n,)) for a in arrays]
    return (*arrays, jnp.float64(rf), jnp.float64(T), jnp.float64(dividend_yield))


def equity_delta_ad(asset_value, asset_vol, debt, rf, T, *, dividend_yield=0.0):  # type: ignore[no-untyped-def]
    """∂E/∂A via JAX autodiff."""
    return _equity_delta_fn(*_broadcast_inputs(asset_value, asset_vol, debt, rf, T, dividend_yield))


def equity_vega_ad(asset_value, asset_vol, debt, rf, T, *, dividend_yield=0.0):  # type: ignore[no-untyped-def]
    """∂E/∂σ_A via JAX autodiff."""
    return _equity_vega_fn(*_broadcast_inputs(asset_value, asset_vol, debt, rf, T, dividend_yield))


def equity_gamma_ad(asset_value, asset_vol, debt, rf, T, *, dividend_yield=0.0):  # type: ignore[no-untyped-def]
    """∂²E/∂A² via JAX autodiff (gradient of the delta)."""
    return _equity_gamma_fn(*_broadcast_inputs(asset_value, asset_vol, debt, rf, T, dividend_yield))


def equity_theta_ad(asset_value, asset_vol, debt, rf, T, *, dividend_yield=0.0):  # type: ignore[no-untyped-def]
    r"""Theta as ``∂E/∂t`` (i.e. ``-∂E/∂T``) — matches the option-pricing
    convention used by :func:`merton.greeks.equity_theta` so the autodiff
    and closed-form columns line up sign-wise."""
    return -_equity_theta_fn(
        *_broadcast_inputs(asset_value, asset_vol, debt, rf, T, dividend_yield)
    )


def equity_rho_ad(asset_value, asset_vol, debt, rf, T, *, dividend_yield=0.0):  # type: ignore[no-untyped-def]
    """∂E/∂r via JAX autodiff."""
    return _equity_rho_fn(*_broadcast_inputs(asset_value, asset_vol, debt, rf, T, dividend_yield))


def pd_leverage_sensitivity_ad(asset_value, asset_vol, debt, rf, T, *, dividend_yield=0.0):  # type: ignore[no-untyped-def]
    """∂PD/∂D via JAX autodiff."""
    return _pd_dleverage_fn(*_broadcast_inputs(asset_value, asset_vol, debt, rf, T, dividend_yield))


def pd_vol_sensitivity_ad(asset_value, asset_vol, debt, rf, T, *, dividend_yield=0.0):  # type: ignore[no-untyped-def]
    """∂PD/∂σ_A via JAX autodiff."""
    return _pd_dvol_fn(*_broadcast_inputs(asset_value, asset_vol, debt, rf, T, dividend_yield))


def pd_rate_sensitivity_ad(asset_value, asset_vol, debt, rf, T, *, dividend_yield=0.0):  # type: ignore[no-untyped-def]
    """∂PD/∂r via JAX autodiff."""
    return _pd_drate_fn(*_broadcast_inputs(asset_value, asset_vol, debt, rf, T, dividend_yield))


__all__ = [
    "equity_delta_ad",
    "equity_gamma_ad",
    "equity_rho_ad",
    "equity_theta_ad",
    "equity_vega_ad",
    "pd_leverage_sensitivity_ad",
    "pd_rate_sensitivity_ad",
    "pd_vol_sensitivity_ad",
]
