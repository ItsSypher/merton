# Sample workbook tour

`merton excel sample-workbook --out=merton-sample.xlsx` produces a
multi-sheet `.xlsx` with worked examples of every formula.

## Sheets

- **Read me** — getting-started instructions inside the workbook.
- **Single firm** — yellow input cells driving `=MERTON.DD`, `MERTON.PD`,
  `MERTON.SPREAD`, `MERTON.ASSET_VALUE`, `MERTON.ASSET_VOL`,
  `MERTON.BLACK_COX`. Includes a 1×6 spilled `MERTON.GREEKS` and a
  `MERTON.PD_TERM` dynamic array.
- **Portfolio** — `MERTON.PORTFOLIO_VAR` with editable PD / LGD / ρ /
  confidence / maturity.
- **Backtest** — toy PD/default series feeding `MERTON.BACKTEST` for
  AUC, Brier, and KS.
- **Function reference** — every formula plus a one-line description.

## Editing inputs

The yellow-filled cells are the only inputs you need to touch. Every
other cell is a formula that recomputes when an input changes. To
experiment with leverage shocks:

1. Open *Single firm*.
2. Change `Debt (default point)` from `35` to e.g. `60`.
3. Watch every output (`DD`, `PD`, `SPREAD`, Greeks, term structure)
   re-evaluate.

## Regenerating the workbook

The sample is generated programmatically by `merton.excel.sample.write_sample_workbook`,
so it's version-controlled as code. If you customise it for your team,
fork that function and re-run `merton excel sample-workbook` to produce
your version.
