"""``Backtest`` orchestrator + ``BacktestResult`` container."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import MertonInputError
from .calibration import CalibrationCurve, calibration_curve
from .metrics import accuracy_ratio, auc, brier, hosmer_lemeshow, ks_statistic
from .roc import ROCCurve, roc_curve


@dataclass(slots=True, frozen=True)
class BacktestResult:
    """Compact summary of a Backtest run."""

    auc: float
    accuracy_ratio: float
    brier: float
    ks: float
    hosmer_lemeshow: tuple[float, float]
    roc: ROCCurve
    calibration: CalibrationCurve
    extra: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        chi2, dof = self.hosmer_lemeshow
        lines = [
            "BacktestResult",
            f"  AUC               : {self.auc:.4f}",
            f"  Accuracy Ratio    : {self.accuracy_ratio:.4f}  (= 2·AUC - 1)",
            f"  Brier score       : {self.brier:.6f}",
            f"  KS statistic      : {self.ks:.4f}",
            f"  Hosmer-Lemeshow χ²: {chi2:.4f}  (dof={int(dof)})",
        ]
        for k, v in self.extra.items():
            lines.append(f"  {k:<17} : {v:.6f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        chi2, dof = self.hosmer_lemeshow
        return {
            "auc": self.auc,
            "accuracy_ratio": self.accuracy_ratio,
            "brier": self.brier,
            "ks": self.ks,
            "hl_chi2": chi2,
            "hl_dof": dof,
            **self.extra,
        }


class Backtest:
    """Compute multiple metrics on a single ``(predictions, defaults)`` pair.

    Examples
    --------
    >>> import numpy as np
    >>> from merton.backtest import Backtest
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> default = rng.binomial(1, 0.1, n)
    >>> pred = np.clip(0.1 + 0.3 * default + rng.normal(0, 0.05, n), 0.001, 0.999)
    >>> bt = Backtest(panel=None, default_col=None, pd_col=None)  # doctest: +SKIP
    """

    def __init__(
        self,
        panel: pd.DataFrame | None = None,
        *,
        default_col: str | None = "default",
        pd_col: str | None = "pd",
        weights_col: str | None = None,
        group_col: str | None = None,
    ) -> None:
        self.panel = panel
        self.default_col = default_col
        self.pd_col = pd_col
        self.weights_col = weights_col
        self.group_col = group_col
        self._extra: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {}

    def add_metric(self, name: str, fn: Callable[[np.ndarray, np.ndarray], float]) -> Backtest:
        """Register an extra metric, applied at :meth:`run` time."""
        self._extra[name] = fn
        return self

    def run(
        self,
        predictions: np.ndarray | None = None,
        defaults: np.ndarray | None = None,
        *,
        n_bins: int = 10,
    ) -> BacktestResult:
        """Compute all metrics. Either pass arrays directly or rely on the
        ``panel`` set at construction."""
        if predictions is None or defaults is None:
            if self.panel is None or self.default_col is None or self.pd_col is None:
                raise MertonInputError(
                    "either pass arrays or initialise Backtest with a panel+col names",
                )
            predictions = self.panel[self.pd_col].to_numpy(dtype=np.float64)
            defaults = self.panel[self.default_col].to_numpy(dtype=np.float64)
        p = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(defaults, dtype=np.float64)
        extra = {name: float(fn(p, y)) for name, fn in self._extra.items()}
        return BacktestResult(
            auc=auc(p, y),
            accuracy_ratio=accuracy_ratio(p, y),
            brier=brier(p, y),
            ks=ks_statistic(p, y),
            hosmer_lemeshow=hosmer_lemeshow(p, y, bins=n_bins),
            roc=roc_curve(p, y),
            calibration=calibration_curve(p, y, bins=n_bins),
            extra=extra,
        )


__all__ = ["Backtest", "BacktestResult"]
