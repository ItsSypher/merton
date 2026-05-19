# JAX acceleration

Install the JAX extra:

```{code-block} bash
uv pip install "merton[jax]"
```

You get three things:

1. The merton kernels run on whatever JAX device backend you've configured
   (CPU, NVIDIA GPU, TPU, Apple Silicon GPU via `jax-metal`).
2. Automatic differentiation of every Greek — handy for validating new
   structural extensions or computing Greeks that don't have a tidy
   closed form.
3. Compositional `vmap` + `jit` for very large panel evaluation.

## Routing inputs through JAX automatically

If you pass JAX arrays to a merton function, the backend dispatch
detects the array namespace and stays on JAX without copying:

```{code-block} python
import jax.numpy as jnp
from merton import distance_to_default

A = jnp.array([100.0, 150.0, 80.0])
dd = distance_to_default(A, 0.25, 60.0, 0.04, 1.0)
type(dd)   # jax.Array
```

You can also force the backend explicitly:

```{code-block} python
distance_to_default(100.0, 0.25, 60.0, 0.04, 1.0, backend="jax")
```

## Autodiff Greeks

`merton.greeks.autodiff` exposes `*_ad` versions of every Greek, computed
by composing `jax.grad`, `jax.vmap`, and `jax.jit`. The arithmetic is
identical to the closed-form path (tests assert agreement within `rtol=1e-5`).

```{code-block} python
import numpy as np
from merton.greeks import autodiff as ad

A = np.array([80.0, 100.0, 150.0])
delta = ad.equity_delta_ad(A, 0.25, 60.0, 0.04, 1.0)
gamma = ad.equity_gamma_ad(A, 0.25, 60.0, 0.04, 1.0)
vega  = ad.equity_vega_ad(A, 0.25, 60.0, 0.04, 1.0)
pd_dD = ad.pd_leverage_sensitivity_ad(A, 0.25, 60.0, 0.04, 1.0)
```

## When to prefer autodiff over the closed forms

Both paths give the same answer for standard BSM Greeks, so the closed
forms in `merton.greeks` are the right default — they're faster and
have no JAX dependency. Reach for `merton.greeks.autodiff` when:

- You're prototyping a new BSM extension whose Greeks aren't derived
  yet — `jax.grad` gives you derivatives for free.
- You need higher-order or mixed partials (e.g. `jax.hessian(...)`).
- You're already on JAX and want zero-copy inter-op.

## Free-threaded Python interaction

On free-threaded Python 3.13t / 3.14t, the JAX backend still uses XLA
threads internally, while merton itself releases the GIL inside Numba
kernels. Mixing the two is supported; `merton doctor` reports the GIL
status so you can confirm the build at a glance.
