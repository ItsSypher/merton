# Installation

`merton` supports Python 3.11 — 3.14 on macOS (Intel + Apple Silicon), Linux
(x86_64 + aarch64), and Windows.

## Minimum (CPU NumPy)

```{code-block} bash
uv pip install merton          # or: pip install merton
```

This installs the core dependencies (NumPy, SciPy, pandas, Numba, structlog,
Pydantic, joblib, Typer) and gives you single-firm fitting + the vectorized
math primitives.

## Optional extras

```{list-table}
:header-rows: 1
:widths: 30 70

* - Goal
  - Install
* - GPU (CUDA 12)
  - ``pip install "merton[gpu]"``
* - JAX autodiff calibration
  - ``pip install "merton[jax]"``
* - Apple Silicon GPU
  - ``pip install "merton[mlx]"``
* - Excel integration
  - ``pip install "merton[excel]"``
* - Plotting (matplotlib + plotly)
  - ``pip install "merton[viz]"``
* - Bayesian MCMC calibration
  - ``pip install "merton[mcmc]"``
* - OpenTelemetry tracing
  - ``pip install "merton[obs]"``
* - Market-data adapters (yfinance, FRED)
  - ``pip install "merton[data]"``
* - Everything
  - ``pip install "merton[all]"``
```

## Verifying the install

```{code-block} bash
python -c "import merton; print(merton.__version__)"
```

You should see a version string. Then trigger a small fit:

```{code-block} python
from merton import Firm, fit
firm = Firm(equity=100, debt_short=20, debt_long=30, equity_vol=0.30)
print(fit(firm).summary())
```
