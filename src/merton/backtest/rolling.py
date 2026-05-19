"""Rolling-window backtester for PD models on a panel of firms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..exceptions import MertonInputError
from .metrics import accuracy_ratio, auc, brier, ks_statistic

if TYPE_CHECKING:
    pass


@dataclass(slots=True, frozen=True)
class RollingBacktestResult:
    """Per-window metric values."""

    window_starts: pd.DatetimeIndex
    auc: np.ndarray
    accuracy_ratio: np.ndarray
    brier: np.ndarray
    ks: np.ndarray

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "window_start": self.window_starts,
                "auc": self.auc,
                "accuracy_ratio": self.accuracy_ratio,
                "brier": self.brier,
                "ks": self.ks,
            }
        )


def rolling_window(
    panel: pd.DataFrame,
    *,
    pd_col: str = "pd",
    default_col: str = "default",
    date_col: str = "date",
    window: str = "252D",
    step: str = "21D",
) -> RollingBacktestResult:
    """Roll a window across ``panel`` and compute AUC/Brier/KS per window.

    Parameters
    ----------
    panel
        Long-form DataFrame with at minimum ``date_col``, ``pd_col``, and
        ``default_col`` columns. ``pd_col`` is the model's predicted PD
        for that (firm, date) and ``default_col`` is the realised 0/1
        default indicator over the next observation period.
    window, step
        Pandas frequency strings; the window slides ``step`` forward at
        each iteration.
    """
    for col in (pd_col, default_col, date_col):
        if col not in panel.columns:
            raise MertonInputError(f"panel is missing column {col!r}")
    df = panel.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    win = pd.Timedelta(window)
    stp = pd.Timedelta(step)
    starts: list[pd.Timestamp] = []
    aucs: list[float] = []
    ars: list[float] = []
    bris: list[float] = []
    kss: list[float] = []
    start = df[date_col].min()
    end = df[date_col].max() - win
    while start <= end:
        block = df[(df[date_col] >= start) & (df[date_col] < start + win)]
        if len(block) >= 10 and block[default_col].nunique() == 2:
            preds = block[pd_col].to_numpy(dtype=np.float64)
            defs = block[default_col].to_numpy(dtype=np.float64)
            try:
                aucs.append(auc(preds, defs))
                ars.append(accuracy_ratio(preds, defs))
                bris.append(brier(preds, defs))
                kss.append(ks_statistic(preds, defs))
                starts.append(start)
            except MertonInputError:
                pass
        start += stp
    return RollingBacktestResult(
        window_starts=pd.DatetimeIndex(starts),
        auc=np.asarray(aucs, dtype=np.float64),
        accuracy_ratio=np.asarray(ars, dtype=np.float64),
        brier=np.asarray(bris, dtype=np.float64),
        ks=np.asarray(kss, dtype=np.float64),
    )


__all__ = ["RollingBacktestResult", "rolling_window"]
