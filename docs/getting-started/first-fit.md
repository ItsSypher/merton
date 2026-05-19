# First fit in five minutes

The simplest call is `merton.fit(firm)`. Let's walk through what each piece
means.

## 1. Build a `Firm`

```{code-block} python
from merton import Firm

firm = Firm(
    equity=100_000_000,        # $100M market cap
    debt_short=20_000_000,     # short-term debt
    debt_long=30_000_000,      # long-term debt
    equity_vol=0.30,           # annualised equity volatility (decimal)
    rf=0.045,                  # risk-free rate
    horizon=1.0,               # T in years
)
```

`Firm` is a frozen dataclass. The default-point formula is KMV
(`short + 0.5 * long`), giving a default threshold of $35M here.

## 2. Fit the model

```{code-block} python
from merton import fit

result = fit(firm, method="vassalou_xing")  # the default
```

The fit returns a `MertonResult` with calibrated asset value, asset volatility,
distance-to-default, and probability of default.

## 3. Inspect the output

```{code-block} python
print(result.summary())
# MertonResult (<firm>, method=vassalou_xing)
#   Distance-to-default     : 3.142
#   Probability of default  : 0.000839
#   Implied spread (LGD=.6) : 5.04 bps
#   Asset value             : 148,599,392.40
#   Asset volatility (σ_A)  : 0.207
#   Default point           : 35,000,000.00
#   Horizon (years)         : 1.0
#   Solver converged        : True (12 iters)
```

## 4. Derived outputs

```{code-block} python
# PD term structure
result.pd_term_structure(horizons=[1/12, 6/12, 1, 3, 5])

# Implied credit spread
result.implied_spread(lgd=0.6)

# Greeks
greeks = result.greeks()
greeks.equity_delta, greeks.pd_dvol

# Physical-measure PD given a Sharpe ratio
result.physical_pd(sharpe_ratio=0.5)

# Export
result.to_pandas()
result.to_excel("merton_results.xlsx")
```

## 5. Calibration alternatives

```{code-block} python
fit(firm, method="naive")           # Bharath-Shumway closed form (fastest)
fit(firm, method="jmr_iterative")   # Jones-Mason-Rosenfeld 2-eq system
fit(firm, method="vassalou_xing")   # iterative MLE (default)
```

When you supply an equity *time series* via `Firm(equity=array_of_prices, ...)`
the Vassalou-Xing calibrator runs in full series mode, iterating until σ_A
converges.
