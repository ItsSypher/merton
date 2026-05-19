# Building a live Excel credit dashboard

This recipe walks through a single-firm credit dashboard built entirely
from `=MERTON.*` formulas — no VBA, no Power Query, no plug-ins beyond
the merton add-in.

## 1. Lay out the inputs

Reserve a small block on the left of the sheet for the yellow input
cells:

```
A2: Equity                B2: 100
A3: Equity vol            B3: 0.30
A4: Debt (default point)  B4: 35
A5: Risk-free rate        B5: 0.04
A6: Horizon (years)       B6: 1
A7: LGD                   B7: 0.6
```

## 2. Core formulas

To the right, lay out the model outputs:

```
D2: DD            E2: =MERTON.DD(B2, B3, B4, B5, B6)
D3: PD            E3: =MERTON.PD(B2, B3, B4, B5, B6)
D4: Spread (bps)  E4: =MERTON.SPREAD(B2, B3, B4, B5, B6, B7)
D5: Asset value   E5: =MERTON.ASSET_VALUE(B2, B3, B4, B5, B6)
D6: Asset vol     E6: =MERTON.ASSET_VOL(B2, B3, B4, B5, B6)
D7: Black-Cox PD  E7: =MERTON.BLACK_COX(B2, B3, B4, B5, B6, 0)
```

## 3. Greeks panel

In `D9:I10`, spill the Greeks:

```
D9: =MERTON.GREEKS(B2, B3, B4, B5, B6)
```

Excel will fill the row labels (D9:I9) and the values (D10:I10)
automatically.

## 4. PD term structure

In `D12:E17`, spill the term structure:

```
D11: Horizons    E11: 0.25  F11: 0.5  G11: 1  H11: 3  I11: 5
D12: =MERTON.PD_TERM(B2, B3, B4, B5, E11:I11)
```

This produces a 2-column dynamic array starting at `D12`.

## 5. Scenario stress test

Use Excel's *What-If Analysis → Data Table* feature to chart `PD` over a
range of equity-vol values:

1. Type the row of σ_E values (e.g. 0.10 → 0.60 in 0.05 steps) along
   row 20 starting at `E20`.
2. Type `=E3` in `D20` (anchor to the PD output).
3. Select `D20:M21` and invoke *Data → What-If Analysis → Data Table*
   with row input cell `B3`.
4. Excel fills `E21:M21` with the PD at each shocked σ_E.
5. Plot the row as a line chart to visualise the stress curve.

## 6. Portfolio view

On a second sheet, list each firm in a column and use
`MERTON.PORTFOLIO_VAR` for the aggregate:

```
A1: Firm    B1: PD      C1: LGD     D1: Exposure
A2: ACME    B2: 0.025   C2: 0.45    D2: 100
A3: WIDGET  B3: 0.012   C3: 0.45    D3: 200
...
F1: Weighted PD            G1: =SUMPRODUCT(B2:B11, D2:D11) / SUM(D2:D11)
F2: Portfolio VaR (IRB)    G2: =MERTON.PORTFOLIO_VAR(G1, 0.45)
```

## 7. Backtest dashboard

Drop two ranges side by side — historical PDs and realised default
indicators — then evaluate AUC / Brier / KS:

```
A1: Historical PD   B1: Realised default (0/1)
A2: ...             B2: ...

D1: AUC      E1: =MERTON.BACKTEST(A2:A1001, B2:B1001, "AUC")
D2: Brier    E2: =MERTON.BACKTEST(A2:A1001, B2:B1001, "Brier")
D3: KS       E3: =MERTON.BACKTEST(A2:A1001, B2:B1001, "KS")
```

## Notes

- All formulas auto-recalculate when you edit any input cell. The Excel
  Office.js runtime calls the local FastAPI server in the background.
- For air-gapped environments, use the classic xlwings UDF path
  (Windows desktop only) — same formulas, no HTTP server.
- Charts and conditional formatting layer over `MERTON.*` outputs
  exactly like they would over any native Excel formula.
