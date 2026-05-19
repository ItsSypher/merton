# merton

[![PyPI version](https://img.shields.io/pypi/v/merton.svg)](https://pypi.org/project/merton/)
[![Python versions](https://img.shields.io/pypi/pyversions/merton.svg)](https://pypi.org/project/merton/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/merton-credit/merton/blob/main/LICENSE)
[![CI](https://github.com/merton-credit/merton/actions/workflows/test.yml/badge.svg)](https://github.com/merton-credit/merton/actions/workflows/test.yml)
[![Docs](https://readthedocs.org/projects/merton/badge/?version=latest)](https://merton.readthedocs.io)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Typed](https://img.shields.io/badge/typed-mypy-blue.svg)](http://mypy-lang.org/)

A production-grade Python package for the **Merton structural credit-risk model**
and its industry-standard descendants (KMV, Black-Cox, Geske, Longstaff-Schwartz,
CreditGrades, Leland-Toft, jump-diffusion, Vasicek portfolio, climate overlays).

## Highlights

- **Vectorized core** — single-firm and panel-scale (10 000+ firms × decades)
  with NumPy by default and optional Numba / JAX / CuPy / MLX backends.
- **Multiple calibration methods** — Vassalou-Xing iterative MLE, Duan
  transformed-data MLE (with survivorship-bias correction), Jones-Mason-Rosenfeld,
  Bharath-Shumway naive, KMV iterative (Crosbie-Bohn).
- **Full extensions library** — Black-Cox first-passage, Geske compound options,
  Longstaff-Schwartz stochastic rates, CreditGrades, Leland-Toft endogenous
  default, Zhou jump-diffusion, hybrid structural/reduced-form, climate overlays.
- **Portfolio engine** — Vasicek single-factor (Basel IRB closed-form),
  Gaussian / t / Clayton / factor copulas, Monte Carlo loss distribution with
  VaR / ES / economic capital.
- **Backtesting harness** — AUC, Brier, KS, accuracy ratio, reliability /
  calibration curves, rolling-window and walk-forward validation.
- **Excel integration** — `=MERTON_DD(equity, σE, debt, rf, T)` and friends,
  available in Excel Web, Excel Mac (M365), and Excel Windows via xlwings Server.
- **Climate stress** — composable `Scenario` framework with packaged NGFS Phase V
  (2024) scenarios (`net_zero_2050`, `delayed_transition`, `current_policies`,
  `fragmented_world`); `ClimateOverlay` wraps any structural model with
  carbon-price paths and sectoral PD multipliers.
- **OpenTelemetry observability** — opt-in via `merton.obs.enable(...)`; pipes
  spans to any OTLP-compatible backend (Datadog, Honeycomb, Tempo, …).
- **Modern Python** — type hints, Pydantic settings, structured logging,
  Apache-2.0 license, cross-platform wheels for Python 3.11-3.14 (incl. free-threaded).

## Quickstart

```bash
uv pip install merton          # or: pip install merton
```

```python
from merton import Firm, fit

firm = Firm(
    equity=100_000_000,        # $100M market cap
    debt_short=20_000_000,
    debt_long=30_000_000,
    equity_vol=0.30,
    rf=0.045,
    horizon=1.0,
)

result = fit(firm, method="vassalou_xing")
print(result.summary())
# MertonResult
#   distance_to_default : 3.142
#   probability_of_default : 0.000839
#   asset_value : 148.6M
#   asset_vol : 0.207
#   implied_spread (LGD=0.6) : 5.0 bps
```

## Excel integration

```bash
merton excel install
merton excel server start --port 8000
```

Then in any Excel workbook (Web / Mac / Windows):

```
=MERTON_DD(B2, B3, B4+B5, B6, B7)        # distance to default
=MERTON_PD(B2, B3, B4+B5, B6, B7)        # probability of default
=MERTON_SPREAD(B2, B3, B4+B5, B6, B7, 0.6)
```

See [docs/excel](https://merton.readthedocs.io/excel) for the full reference.

## Performance

- Single-firm fit: <50 ms
- 10 000-firm × 10-year daily panel: <60 s on 8 cores (Numba backend)
- 100 000-sim portfolio VaR: <30 s with GPU (CuPy)

## Documentation

Full docs at <https://merton.readthedocs.io> including:

- Theory deep-dives (Merton 1974, KMV, Black-Cox, Geske, …)
- Executable tutorials (AAPL time series, panel backtest, portfolio VaR,
  climate stress test, GPU acceleration, Excel integration)
- API reference
- Performance benchmarks and backend-selection guide

## Installation matrix

| Goal | Command |
|---|---|
| Minimum (CPU NumPy) | `pip install merton` |
| Numba JIT (default fast path) | `pip install merton` (Numba is a hard dep) |
| GPU acceleration | `pip install "merton[gpu]"` (requires CUDA 12) |
| JAX autodiff calibration | `pip install "merton[jax]"` |
| Apple Silicon GPU | `pip install "merton[mlx]"` |
| Excel integration | `pip install "merton[excel]"` |
| Visualization (matplotlib + plotly) | `pip install "merton[viz]"` |
| Bayesian MCMC calibration | `pip install "merton[mcmc]"` |
| OpenTelemetry tracing | `pip install "merton[obs]"` |
| Everything | `pip install "merton[all]"` |

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Citing

If you use `merton` in academic work, please cite via the
[CITATION.cff](CITATION.cff) file or the package's Zenodo DOI (assigned on first
tagged release).
