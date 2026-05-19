---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Tutorial 1 — single firm, AAPL-style walk-through

This tutorial fits the Merton model to an Apple-like firm. We use a
**synthetic** equity time series (so the docs build is hermetic), but the
exact same code works on a real AAPL series fetched via
`Firm.from_yfinance("AAPL")` once you `pip install "merton[data]"`.

## Set up

```{code-cell} python
import numpy as np

import merton
from merton import Firm, MertonModel, fit

print("merton version:", merton.__version__)
```

## Build a synthetic AAPL series

We simulate one year of daily prices under geometric Brownian motion with a
typical large-cap volatility of ~25 % and a small upward drift.

```{code-cell} python
rng = np.random.default_rng(2026)

n_days = 252
sigma_E = 0.25
mu_E = 0.10
dt = 1.0 / 252.0
log_returns = rng.normal((mu_E - 0.5 * sigma_E**2) * dt, sigma_E * np.sqrt(dt), size=n_days)
equity_path = 3.0e12 * np.exp(np.cumsum(log_returns))  # starts ≈ $3T market cap

firm = Firm(
    equity=equity_path,
    debt_short=20e9,
    debt_long=90e9,
    rf=0.045,
    dividend_yield=0.006,
    horizon=1.0,
    ticker="AAPL_SYNTH",
)
print("equity range:", f"${equity_path.min()/1e12:.2f}T", "→", f"${equity_path.max()/1e12:.2f}T")
print("default point (KMV):", f"${float(firm.default_point_value())/1e9:.1f}B")
```

## Three calibration methods side-by-side

```{code-cell} python
snapshot_firm = firm.replace(equity=float(equity_path[-1]), equity_vol=sigma_E)

methods = ["naive", "jmr_iterative", "vassalou_xing"]
print(f"{'method':<18}{'DD':>10}{'PD':>14}{'σ_A':>10}")
print("-" * 52)
for m in methods:
    result = fit(snapshot_firm, method=m)
    print(f"{m:<18}{result.dd:>10.4f}{result.pd:>14.6e}{result.asset_vol:>10.4f}")
```

## Full Duan MLE on the equity series

```{code-cell} python
result = MertonModel(method="duan_mle").fit(firm)
print(result.summary())
```

The Duan MLE also returns asymptotic standard errors, which we can turn into
confidence intervals on any quantity of interest:

```{code-cell} python
ci = result.confidence_interval(level=0.95, method="asymptotic")
for name, interval in ci.items():
    print(f"  {name:<14} 95% CI = [{interval.lower:+.4g}, {interval.upper:+.4g}]")
```

## Bootstrapped confidence intervals

When the calibrator doesn't return a Hessian (or we want a CI that doesn't
assume MLE asymptotics), the model can run a block bootstrap on the equity
series:

```{code-cell} python
bs_result = MertonModel(
    method="vassalou_xing",
    n_bootstrap=100,
    block_length=20,
    random_state=42,
).fit(firm)

bs_ci = bs_result.confidence_interval(level=0.95, method="bootstrap")
for name, interval in bs_ci.items():
    print(f"  {name:<14} bootstrap CI = [{interval.lower:+.4g}, {interval.upper:+.4g}]")
```

## PD term structure

```{code-cell} python
result.pd_term_structure(horizons=[1/12, 3/12, 6/12, 1, 3, 5])
```

## Greeks at the calibrated operating point

```{code-cell} python
g = result.greeks()
print(f"  equity Δ      : {g.equity_delta:.6f}")
print(f"  equity Vega   : {g.equity_vega:.6e}")
print(f"  equity Γ      : {g.equity_gamma:.6e}")
print(f"  ∂PD/∂L        : {g.pd_dleverage:.6e}")
print(f"  ∂PD/∂σ_A      : {g.pd_dvol:.6e}")
print(f"  ∂PD/∂r        : {g.pd_drate:.6e}")
```

## What changes with real data?

Swap the synthetic series for a real AAPL pull:

```python
firm = Firm.from_yfinance("AAPL", lookback="2y")  # needs merton[data]
```

…and re-run any of the cells above. The package interface stays identical.
