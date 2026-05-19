"""``merton fit`` — fit a single firm or a panel from a CSV/Parquet input."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

console = Console()


def run(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Input file. CSV (single row → single-firm) or Parquet (panel).",
    ),
    method: str = typer.Option(
        "vassalou_xing",
        "--method",
        "-m",
        help="Calibration method: vassalou_xing, jmr_iterative, naive, kmv_iterative, duan_mle.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output file (.csv, .parquet, .xlsx, .json). If omitted, prints summary to stdout.",
    ),
    n_jobs: int = typer.Option(-1, "--n-jobs", "-j", help="joblib n_jobs for panel mode."),
    horizon: float = typer.Option(1.0, "--horizon", "-T", help="Horizon in years."),
) -> None:
    """Fit a single firm (one-row CSV) or a panel (Parquet / multi-row CSV)."""
    import pandas as pd

    from ... import Firm, MertonModel
    from ...batch import batch_fit

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    if len(df) == 1 and "ticker" not in df.columns:
        row = df.iloc[0]
        firm = Firm(
            equity=float(row["equity"]),
            debt_short=float(row["debt_short"]),
            debt_long=float(row["debt_long"]),
            equity_vol=float(row["equity_vol"]) if "equity_vol" in row else None,
            rf=float(row.get("rf", 0.04)),
            horizon=horizon,
            ticker=str(row.get("ticker")) if "ticker" in row else None,
        )
        result = MertonModel(method=method).fit(firm)
        if out is None:
            console.print(result.summary())
            return
        if out.suffix.lower() in {".csv", ".xlsx", ".parquet", ".json"}:
            result.to_pandas().to_csv(out, index=False) if out.suffix.lower() == ".csv" else None
            if out.suffix.lower() == ".xlsx":
                result.to_excel(str(out))
            if out.suffix.lower() == ".parquet":
                result.to_pandas().to_parquet(out)
            if out.suffix.lower() == ".json":
                out.write_text(__import__("json").dumps(result.to_dict(), default=str, indent=2))
        console.print(f"[green]wrote[/] {out}")
        return

    # Panel mode.
    panel = batch_fit(df, method=method, n_jobs=n_jobs, horizon=horizon, progress=True)
    if out is None:
        console.print(panel.head(20).to_string(index=False))
        return
    suffix = out.suffix.lower()
    if suffix == ".csv":
        panel.to_csv(out, index=False)
    elif suffix == ".parquet":
        panel.to_parquet(out)
    elif suffix == ".xlsx":
        panel.to_excel(out, index=False)
    elif suffix == ".json":
        panel.to_json(out, orient="records", indent=2)
    else:
        raise typer.BadParameter(f"unsupported output suffix {suffix!r}")
    console.print(f"[green]wrote[/] {out} ({len(panel)} rows)")
