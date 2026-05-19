# Apple Silicon notes

`merton` is built and tested on Apple Silicon Macs (M1/M2/M3/M4) as a
first-class target. Wheels are published for `macos-14 arm64` in the
release matrix, so `pip install merton` Just Works on every Apple
Silicon laptop running Python 3.11+.

## MLX (Metal GPU)

Install the `mlx` extra to route panel math to the GPU:

```{code-block} bash
uv pip install "merton[mlx]"
```

MLX uses a **unified-memory** model: arrays live in shared memory that the
CPU and GPU both see. There's no host-to-device copy when handing data
between code paths, which is a big win versus the CUDA model. The Metal
kernels are routed via `mlx.core.erf` for the normal CDF.

```{code-block} python
import mlx.core as mx
from merton import distance_to_default

A = mx.array([100.0, 200.0, 300.0])
dd = distance_to_default(A, 0.25, 60.0, 0.04, 1.0)  # stays on MLX
```

## Free-threaded Python

Apple Silicon laptops typically have 8-12 performance cores. Combined with
free-threaded Python 3.13t / 3.14t, panel calibration scales linearly to
the core count. See {doc}`free-threaded`.

## Native universal2 wheels?

We publish **arm64-only** Apple Silicon wheels (no universal2). Intel
Macs use the separate `macos-13 x86_64` wheel. The reason is wheel size:
universal2 doubles the bundled Numba cache for no real benefit (Apple
stopped shipping Intel laptops in 2022).
