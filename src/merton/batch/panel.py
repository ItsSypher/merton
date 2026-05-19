"""``batch_fit`` — calibrate Merton over a panel of firms in parallel."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..core.firm import Firm
from ..core.model import MertonModel
from ..core.panel import FirmPanel
from .dispatch import parallel_map

if TYPE_CHECKING:
    pass


def _panelify(df: Any, mapping: dict[str, str] | None) -> tuple[FirmPanel, str]:
    """Coerce input to a FirmPanel; return (panel, output_kind)."""
    if isinstance(df, FirmPanel):
        return df, "panel"
    if isinstance(df, pd.DataFrame):
        return FirmPanel.from_pandas(df, mapping=mapping), "pandas"
    try:
        import polars as pl  # type: ignore[import-not-found]
    except ImportError:
        pl = None
    if pl is not None and isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        return FirmPanel.from_polars(df, mapping=mapping), "polars"
    import pyarrow as pa

    if isinstance(df, pa.Table):
        return FirmPanel.from_arrow(df), "arrow"
    raise TypeError(f"batch_fit cannot accept {type(df).__name__}")


def _format_output(rows: list[dict[str, Any]], kind: str) -> Any:
    df = pd.DataFrame(rows)
    if kind == "pandas":
        return df
    if kind == "polars":  # pragma: no cover - optional
        import polars as pl  # type: ignore[import-not-found]

        return pl.from_pandas(df)
    if kind == "arrow":  # pragma: no cover - optional
        import pyarrow as pa

        return pa.Table.from_pandas(df, preserve_index=False)
    return df  # plain FirmPanel input → pandas DataFrame output


def batch_fit(
    df: Any,
    *,
    method: str = "vassalou_xing",
    mapping: dict[str, str] | None = None,
    n_jobs: int = -1,
    dispatch: str = "joblib",
    chunk_size: int | None = None,
    progress: bool = False,
    on_error: str = "warn",
    horizon: float | None = None,
    **fit_kwargs: Any,
) -> Any:
    """Fit a Merton model to every firm in a panel.

    Parameters
    ----------
    df
        Either a :class:`FirmPanel`, pandas DataFrame, polars DataFrame, or
        pyarrow Table containing one row per (firm, snapshot). Required columns:
        ``equity``, ``debt_short``, ``debt_long``. Optional: ``equity_vol``,
        ``rf``, ``dividend_yield``, ``horizon``, ``ticker``.
    method
        Calibration method (any name in
        :func:`merton.calibration.available_methods`).
    mapping
        Optional column-rename map applied before validation.
    n_jobs
        joblib workers. ``-1`` = all logical cores.
    dispatch
        ``"joblib"`` (default), ``"sequential"``, ``"dask"``, or ``"ray"``.
    chunk_size
        Chunk size for the dispatch. Default: ``max(1, len(panel)//n_jobs)``.
        (Currently advisory — joblib chooses chunk sizes automatically.)
    progress
        Render a Rich progress bar.
    on_error
        ``"warn"`` (default) emits a warning and continues with NaNs.
        ``"raise"`` propagates the first exception. ``"skip"`` drops the row.
    horizon
        Override the horizon for every firm (handy when the input panel
        doesn't carry one).
    **fit_kwargs
        Forwarded to :class:`MertonModel` (e.g. ``tol``, ``max_iter``,
        ``physical_measure``, ``sharpe_ratio``, ``n_bootstrap``).
    """
    panel, out_kind = _panelify(df, mapping)
    if len(panel) == 0:
        return _format_output([], out_kind)

    model_template = {"method": method, **fit_kwargs}

    def _fit_one(firm: Firm) -> dict[str, Any]:
        if horizon is not None:
            firm = firm.replace(horizon=horizon)
        try:
            res = MertonModel(**model_template).fit(firm)
        except Exception as err:
            if on_error == "raise":
                raise
            if on_error == "warn":
                warnings.warn(
                    f"batch_fit: {type(err).__name__}: {err} (firm={firm.ticker})",
                    stacklevel=3,
                )
            return {
                "ticker": firm.ticker,
                "dd": np.nan,
                "pd": np.nan,
                "asset_value": np.nan,
                "asset_vol": np.nan,
                "method": method,
                "converged": False,
            }
        return res.to_dict()

    firms = list(panel)
    results = parallel_map(
        _fit_one,
        firms,
        dispatch=dispatch,
        n_jobs=n_jobs,
        progress=progress,
        description=f"merton.batch_fit({method})",
    )
    rows = [r for r in results if not (on_error == "skip" and r.get("converged") is False)]
    return _format_output(rows, out_kind)


__all__ = ["batch_fit"]
