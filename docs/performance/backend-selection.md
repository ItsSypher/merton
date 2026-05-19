# Backend selection guide

`merton` dispatches the heavy math kernels to one of five backends:

| Backend | When to use | Install |
|---|---|---|
| `numpy` | Default for small arrays (< 256 elements) and laptops without Numba caches. | core (always available) |
| `numba` | CPU default. JIT-compiled SIMD; releases the GIL. | core (always available) |
| `cupy` | NVIDIA GPU panels (100k+ firms) and large Monte Carlo. | `pip install "merton[gpu]"` (requires CUDA 12) |
| `jax` | Autodiff Greeks; cross-device portability. | `pip install "merton[jax]"` |
| `mlx` | Apple Silicon GPU via Metal; unified-memory model means zero-copy from NumPy. | `pip install "merton[mlx]"` |

## Resolution order

For every public numerical function, the dispatcher picks a backend in this
order:

1. **Explicit** `backend="..."` kwarg.
2. **Input-array namespace**: a `cupy.ndarray` stays on the GPU,
   `jax.Array` stays on JAX, `mlx.array` stays on MLX — no host transfers.
3. **`MERTON_BACKEND` env var** (`numpy` | `numba` | `cupy` | `jax` | `mlx`).
4. **Size heuristic**: arrays larger than `numba_threshold` (default 256)
   take the Numba path on CPU; smaller stay on NumPy.
5. **Fallback** when the requested backend isn't installed — we emit a
   `MertonBackendFallbackWarning` and pick the closest available kernel.

## How fast is each backend?

Order of magnitude (Apple Silicon M-class):

- `numpy` < `numba` < `mlx` ≈ `cupy` < `jax`

`numpy` is the simplest; `numba` adds SIMD + GIL release;
`mlx`/`cupy`/`jax` add device parallelism but pay launch-latency on small
arrays. The break-even point is usually around 1 000-10 000 elements on
single-firm work and "always faster" once you hit 100k+ firms.

## Code that adapts to whatever backend the user has

```{code-block} python
import jax.numpy as jnp        # or cupy as cp, or mlx.core as mx
from merton import distance_to_default

# Pass JAX arrays in — output stays on JAX:
arr = jnp.array([100.0, 200.0, 300.0])
dd = distance_to_default(arr, 0.25, 60.0, 0.04, 1.0)
# type(dd) is jax.Array
```

## Forcing a backend

```{code-block} python
distance_to_default(100.0, 0.25, 60.0, 0.04, 1.0, backend="numpy")
# Or globally:
import os; os.environ["MERTON_BACKEND"] = "numba"
```

Confirm what's installed with `merton doctor`.
