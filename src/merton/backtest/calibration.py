"""Calibration / reliability curves for PD models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._typing import ArrayLike, FloatArray
from ..exceptions import MertonInputError

if TYPE_CHECKING:
    import matplotlib.axes


@dataclass(slots=True, frozen=True)
class CalibrationCurve:
    """Output of :func:`calibration_curve`."""

    mean_predicted: FloatArray
    fraction_positives: FloatArray
    bin_counts: FloatArray


def calibration_curve(
    predictions: ArrayLike,
    defaults: ArrayLike,
    *,
    bins: int = 10,
    strategy: str = "quantile",
) -> CalibrationCurve:
    """Reliability / calibration curve.

    Parameters
    ----------
    bins
        Number of bins.
    strategy
        ``"quantile"`` (default) puts equal counts in each bin;
        ``"uniform"`` uses equal-width bins on ``[0, 1]``.
    """
    p = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(defaults, dtype=np.float64)
    if p.shape != y.shape:
        raise MertonInputError("predictions and defaults must have the same shape")
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, bins + 1)
    else:
        raise MertonInputError(
            f"unknown strategy {strategy!r}",
            suggested_fix="Choose 'quantile' or 'uniform'.",
        )
    # Ensure edges are strictly increasing.
    edges[-1] += 1e-12
    bin_idx = np.digitize(p, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, bins - 1)
    mean_pred = np.zeros(bins)
    frac_pos = np.zeros(bins)
    counts = np.zeros(bins)
    for b in range(bins):
        mask = bin_idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            mean_pred[b] = (edges[b] + edges[b + 1]) / 2
            frac_pos[b] = np.nan
        else:
            mean_pred[b] = float(p[mask].mean())
            frac_pos[b] = float(y[mask].mean())
        counts[b] = cnt
    return CalibrationCurve(
        mean_predicted=mean_pred,
        fraction_positives=frac_pos,
        bin_counts=counts,
    )


def calibration_plot(
    predictions: ArrayLike,
    defaults: ArrayLike,
    *,
    bins: int = 10,
    strategy: str = "quantile",
    ax: matplotlib.axes.Axes | None = None,
) -> Any:
    """Render a calibration plot. Returns the matplotlib Axes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:  # pragma: no cover - optional dep
        raise ImportError(
            'merton.backtest.calibration_plot requires matplotlib: `pip install "merton[viz]"`.'
        ) from err
    cc = calibration_curve(predictions, defaults, bins=bins, strategy=strategy)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    ax.plot(cc.mean_predicted, cc.fraction_positives, marker="o", label="model")
    ax.set_xlabel("mean predicted PD")
    ax.set_ylabel("observed default rate")
    ax.legend(loc="best")
    return ax


__all__ = ["CalibrationCurve", "calibration_curve", "calibration_plot"]
