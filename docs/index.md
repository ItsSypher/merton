# merton

A production-grade Python package for the **Merton structural credit-risk
model** and its descendants.

```{toctree}
:hidden:
:caption: Getting started

getting-started/install
getting-started/first-fit
getting-started/excel-quickstart
```

```{toctree}
:hidden:
:caption: Theory

theory/merton-1974
theory/kmv
theory/calibration
theory/extensions/black-cox
theory/extensions/geske
theory/extensions/creditgrades
theory/extensions/leland-toft
theory/extensions/jump-diffusion
theory/extensions/longstaff-schwartz
theory/portfolio/vasicek
theory/portfolio/copulas
theory/metrics
theory/bayesian-mcmc
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/01_single_firm_aapl
```

```{toctree}
:hidden:
:caption: Excel

excel/installation
excel/functions
excel/sample-workbook
```

```{toctree}
:hidden:
:caption: Cookbook

cookbook/panel-fitting
cookbook/jax-acceleration
cookbook/large-panels
cookbook/yfinance
cookbook/bloomberg
cookbook/excel-dashboard
```

```{toctree}
:hidden:
:caption: Performance

performance/benchmarks
performance/backend-selection
performance/apple-silicon
performance/free-threaded
```

```{toctree}
:hidden:
:caption: API reference

api/merton/index
```

```{toctree}
:hidden:
:caption: Project

changelog
references
```

## What is the Merton model?

The Merton (1974) structural credit-risk model treats a firm's equity as a
European call option on the firm's underlying assets. Default occurs when the
asset value falls below a debt threshold at maturity. The package implements:

- Single-firm calibration (Vassalou-Xing, Duan MLE, Jones-Mason-Rosenfeld,
  Bharath-Shumway naive, KMV-iterative).
- Vectorized math primitives — `distance_to_default`, `prob_of_default`,
  `implied_credit_spread`, `term_structure_pd`.
- Equity Greeks and PD sensitivities in closed form (plus optional JAX autodiff).
- Structural extensions (Black-Cox, Geske compound options, Longstaff-Schwartz,
  CreditGrades, Leland-Toft, Zhou jump-diffusion, hybrid, climate overlays).
- Portfolio engine (Vasicek single-factor, copula simulations, VaR / ES / EC).
- Backtest harness (AUC, Brier, KS, calibration plots, rolling-window).
- Excel formulas via xlwings Server (`=MERTON_DD(…)`, …).

## Why a new package?

Existing Python options ship single-firm Merton math but lack ergonomics,
calibration variety, extensions, portfolio modelling, and Excel integration.
`merton` is built to fill all of those gaps in a single coherent package.

## Quickstart

```{code-block} python
from merton import Firm, fit

firm = Firm(
    equity=100_000_000,
    debt_short=20_000_000,
    debt_long=30_000_000,
    equity_vol=0.30,
    rf=0.045,
    horizon=1.0,
)
result = fit(firm, method="vassalou_xing")
print(result.summary())
```

See {doc}`getting-started/install` and {doc}`getting-started/first-fit` to
get started.
