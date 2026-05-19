# Excel integration quickstart

`merton[excel]` ships an xlwings Server + Office.js add-in so users can call
Merton functions as **native Excel formulas** in Web, Mac, and Windows Excel.

> Status: ships in Phase 0.6 (see {doc}`/changelog`). This page is a preview.

## Install the add-in

```{code-block} bash
pip install "merton[excel]"
merton excel install
```

## Start the local server

```{code-block} bash
merton excel server start --port 8000
```

Leave the terminal running — Excel calls it for every recalculation.

## Use it

| Formula | Returns |
|---|---|
| `=MERTON_DD(E, σE, D, r, T)` | distance to default |
| `=MERTON_PD(E, σE, D, r, T)` | probability of default |
| `=MERTON_SPREAD(E, σE, D, r, T, LGD)` | implied credit spread (bps) |
| `=MERTON_ASSET_VALUE(E, σE, D, r, T)` | inverted asset value |
| `=MERTON_ASSET_VOL(E, σE, D, r, T)` | inverted σ_A |
| `=MERTON_PD_TERM(E, σE, D, r, {horizons…})` | PD term structure (spilled array) |
| `=MERTON_GREEKS(E, σE, D, r, T)` | 6-cell array of Greeks |
| `=MERTON_BACKTEST(pd_range, defaults_range, "AUC")` | model-validation metrics |

## Stop the server

```{code-block} bash
merton excel server stop
```
