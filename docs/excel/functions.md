# Excel function reference

Every formula lives in the `MERTON` namespace. After sideloading the
add-in (`merton excel install`) and starting the server
(`merton excel server start`), Excel autocomplete will suggest each
formula when you type `=MERTON.`.

## Single-firm

```{list-table}
:header-rows: 1

* - Formula
  - Returns
* - `=MERTON.DD(equity, σE, debt, rf, T)`
  - Distance to default
* - `=MERTON.PD(equity, σE, debt, rf, T)`
  - Probability of default at horizon `T`
* - `=MERTON.SPREAD(equity, σE, debt, rf, T, LGD)`
  - Implied credit spread (basis points)
* - `=MERTON.ASSET_VALUE(equity, σE, debt, rf, T)`
  - Inverted firm asset value `A`
* - `=MERTON.ASSET_VOL(equity, σE, debt, rf, T)`
  - Inverted asset volatility `σ_A`
* - `=MERTON.GREEKS(equity, σE, debt, rf, T)`
  - 1×6 array: Δ, Γ, Vega, Θ, ρ, ∂PD/∂L
* - `=MERTON.PD_TERM(equity, σE, debt, rf, {horizons…})`
  - PD term structure (2-column dynamic array)
* - `=MERTON.BLACK_COX(equity, σE, debt, rf, T, γ)`
  - Black-Cox first-passage PD
```

## Portfolio

```{list-table}
:header-rows: 1

* - Formula
  - Returns
* - `=MERTON.PORTFOLIO_VAR(PD, LGD, ρ?, α?, M?)`
  - Basel-IRB Vasicek VaR (Loss / Exposure)
```

When `ρ` is omitted (or blank), the formula uses the Basel-prescribed
asset correlation derived from the PD.

## Backtest

```{list-table}
:header-rows: 1

* - Formula
  - Returns
* - `=MERTON.BACKTEST(pd_range, defaults_range, "AUC")`
  - AUC of the model
* - `=MERTON.BACKTEST(pd_range, defaults_range, "Brier")`
  - Brier score
* - `=MERTON.BACKTEST(pd_range, defaults_range, "KS")`
  - Kolmogorov-Smirnov statistic
* - `=MERTON.BACKTEST(pd_range, defaults_range, "AccuracyRatio")`
  - Accuracy ratio (Gini coefficient)
```

## Argument conventions

- `equity`, `debt` — currency units (whatever's consistent across your
  sheet).
- `σE`, `σ_A`, `LGD`, `α`, `ρ` — decimals (use `0.30`, not `30%`).
- `rf` — annualised continuously-compounded decimal.
- `T` — years (e.g. `0.5` for six months, `5` for five years).
- Array arguments — Excel ranges (e.g. `A2:A11`) or inline arrays
  (e.g. `{0.25, 0.5, 1, 3, 5}`).

Hover any function name in Excel's formula wizard for inline parameter
descriptions.
