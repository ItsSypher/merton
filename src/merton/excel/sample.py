"""Programmatic sample workbook for the Excel integration.

Generates a multi-sheet ``.xlsx`` with worked examples of every
``=MERTON_*`` formula. Users open the workbook *after* sideloading the
add-in (``merton excel install``); the formulas then evaluate against
their local server.

The function uses :mod:`openpyxl` so the file is fully native Excel —
formulas, formatting, headers, the lot — and works in Excel for the web
without recalculation surprises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .functions import EXCEL_FUNCTIONS


def _set_header(ws: Any, row: int, labels: list[str]) -> None:
    from openpyxl.styles import Font

    bold = Font(bold=True)
    for col, label in enumerate(labels, start=1):
        cell = ws.cell(row=row, column=col, value=label)
        cell.font = bold


def write_sample_workbook(path: str | Path) -> Path:
    """Write a worked-example workbook to ``path`` and return the path."""
    try:
        import openpyxl  # type: ignore[import-not-found]
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError as err:  # pragma: no cover - excel extra
        raise ImportError("merton.excel.sample requires merton[excel]: install openpyxl.") from err

    wb = openpyxl.Workbook()
    ws_intro = wb.active
    assert ws_intro is not None
    ws_intro.title = "Read me"
    ws_intro["A1"] = "merton — Excel sample workbook"
    ws_intro["A1"].font = Font(bold=True, size=16)
    ws_intro["A3"] = (
        "Sideload the merton add-in first: run `merton excel install` and then "
        "start the server with `merton excel server start`."
    )
    ws_intro["A4"] = (
        "All formulas below live in the MERTON namespace, e.g. =MERTON.DD(...). "
        "Excel will autocomplete them once the add-in is loaded."
    )
    ws_intro["A5"] = "Replace placeholder values in the yellow cells to see live results."
    ws_intro.column_dimensions["A"].width = 100

    # --- Single firm tab ---------------------------------------------------
    ws = wb.create_sheet("Single firm")
    _set_header(ws, 1, ["Field", "Value", "Description"])
    inputs = [
        ("Equity (market cap)", 100.0, "Market value of equity (currency units)"),
        ("Equity vol (σ_E)", 0.30, "Annualised equity volatility (decimal)"),
        ("Debt (default point)", 35.0, "Short-term + 0.5·long-term debt (KMV)"),
        ("Risk-free rate", 0.04, "Annualised decimal"),
        ("Horizon (years)", 1.0, "Time horizon for DD/PD"),
        ("LGD", 0.6, "Loss given default"),
    ]
    highlight = PatternFill(start_color="FFF7B2", end_color="FFF7B2", fill_type="solid")
    for i, (label, value, description) in enumerate(inputs, start=2):
        ws.cell(row=i, column=1, value=label)
        cell = ws.cell(row=i, column=2, value=value)
        cell.fill = highlight
        ws.cell(row=i, column=3, value=description)
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 50

    # Outputs section with formulas.
    ws.cell(row=9, column=1, value="Outputs").font = Font(bold=True)
    _set_header(ws, 10, ["Formula", "Value"])
    formulas = [
        ("Distance to default", "=MERTON.DD(B2, B3, B4, B5, B6)"),
        ("Probability of default", "=MERTON.PD(B2, B3, B4, B5, B6)"),
        ("Implied spread (bps)", "=MERTON.SPREAD(B2, B3, B4, B5, B6, B7)"),
        ("Asset value", "=MERTON.ASSET_VALUE(B2, B3, B4, B5, B6)"),
        ("Asset volatility", "=MERTON.ASSET_VOL(B2, B3, B4, B5, B6)"),
        ("Black-Cox PD", "=MERTON.BLACK_COX(B2, B3, B4, B5, B6, 0)"),
    ]
    for i, (label, formula) in enumerate(formulas, start=11):
        ws.cell(row=i, column=1, value=label)
        c = ws.cell(row=i, column=2, value=formula)
        c.alignment = Alignment(horizontal="left")

    # Greeks block.
    ws.cell(row=18, column=1, value="Greeks (spilled 1×6 array)").font = Font(bold=True)
    ws.cell(row=19, column=1, value="=MERTON.GREEKS(B2, B3, B4, B5, B6)").alignment = Alignment(
        horizontal="left"
    )

    # Term structure block.
    ws.cell(row=22, column=1, value="PD term structure").font = Font(bold=True)
    ws.cell(row=23, column=1, value="Horizons:")
    ws.cell(row=23, column=2, value=0.25)
    ws.cell(row=23, column=3, value=0.5)
    ws.cell(row=23, column=4, value=1.0)
    ws.cell(row=23, column=5, value=3.0)
    ws.cell(row=23, column=6, value=5.0)
    ws.cell(
        row=24, column=1, value="=MERTON.PD_TERM(B2, B3, B4, B5, B23:F23)"
    ).alignment = Alignment(horizontal="left")

    # --- Portfolio tab ----------------------------------------------------
    ws_pf = wb.create_sheet("Portfolio")
    _set_header(ws_pf, 1, ["Field", "Value"])
    pf_inputs = [
        ("PD", 0.02),
        ("LGD", 0.45),
        ("Asset correlation (blank = Basel)", None),
        ("Confidence", 0.999),
        ("Maturity (years)", 2.5),
    ]
    for i, (label, value) in enumerate(pf_inputs, start=2):
        ws_pf.cell(row=i, column=1, value=label)
        cell = ws_pf.cell(row=i, column=2, value=value)
        cell.fill = highlight
    ws_pf.cell(row=8, column=1, value="Basel-IRB Vasicek VaR (loss/exposure)").font = Font(
        bold=True
    )
    ws_pf.cell(
        row=9, column=1, value="=MERTON.PORTFOLIO_VAR(B2, B3, B4, B5, B6)"
    ).alignment = Alignment(horizontal="left")
    ws_pf.column_dimensions["A"].width = 38

    # --- Backtest tab -----------------------------------------------------
    ws_bt = wb.create_sheet("Backtest")
    _set_header(ws_bt, 1, ["PD prediction", "Default (0/1)"])
    # Synthetic toy series.
    sample_preds = [0.02, 0.05, 0.10, 0.15, 0.40, 0.50, 0.08, 0.03, 0.20, 0.45]
    sample_defs = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
    for i, (p, d) in enumerate(zip(sample_preds, sample_defs, strict=True), start=2):
        ws_bt.cell(row=i, column=1, value=p)
        ws_bt.cell(row=i, column=2, value=d)
    ws_bt.cell(row=14, column=1, value="AUC")
    ws_bt.cell(
        row=14, column=2, value='=MERTON.BACKTEST(A2:A11, B2:B11, "AUC")'
    ).alignment = Alignment(horizontal="left")
    ws_bt.cell(row=15, column=1, value="Brier")
    ws_bt.cell(
        row=15, column=2, value='=MERTON.BACKTEST(A2:A11, B2:B11, "Brier")'
    ).alignment = Alignment(horizontal="left")
    ws_bt.cell(row=16, column=1, value="KS")
    ws_bt.cell(
        row=16, column=2, value='=MERTON.BACKTEST(A2:A11, B2:B11, "KS")'
    ).alignment = Alignment(horizontal="left")

    # --- Function reference tab ------------------------------------------
    ws_ref = wb.create_sheet("Function reference")
    _set_header(ws_ref, 1, ["Formula", "Description"])
    for i, (excel_name, description, _fn) in enumerate(EXCEL_FUNCTIONS, start=2):
        ws_ref.cell(row=i, column=1, value=excel_name)
        ws_ref.cell(row=i, column=2, value=description)
    ws_ref.column_dimensions["A"].width = 26
    ws_ref.column_dimensions["B"].width = 70

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


__all__ = ["write_sample_workbook"]
