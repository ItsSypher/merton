"""ROC-curve construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._typing import ArrayLike, FloatArray
from ..exceptions import MertonInputError


@dataclass(slots=True, frozen=True)
class ROCCurve:
    """Cached ROC curve: false-positive rate, true-positive rate, thresholds."""

    fpr: FloatArray
    tpr: FloatArray
    thresholds: FloatArray

    def auc(self) -> float:
        # Trapezoidal area along (fpr, tpr). ``np.trapezoid`` is the NumPy
        # 2.x name; fall back to ``np.trapz`` for older installs.
        trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]  # noqa: NPY201
        return float(trapezoid(self.tpr, self.fpr))


def roc_curve(predictions: ArrayLike, defaults: ArrayLike) -> ROCCurve:
    """Compute ROC curve via cumulative true/false positives."""
    p = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(defaults, dtype=np.float64)
    if p.shape != y.shape:
        raise MertonInputError("predictions and defaults must have the same shape")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise MertonInputError("defaults must be 0/1 indicators")
    # Sort descending by predicted PD.
    order = np.argsort(-p)
    y_sorted = y[order]
    p_sorted = p[order]
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise MertonInputError("ROC undefined when only one class is present")
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    # Drop duplicate thresholds (the canonical scikit-learn convention).
    distinct = np.r_[np.diff(p_sorted) != 0, [True]]
    tps = np.r_[0.0, tps[distinct]]
    fps = np.r_[0.0, fps[distinct]]
    thr = np.r_[np.inf, p_sorted[distinct]]
    return ROCCurve(
        fpr=fps / n_neg,
        tpr=tps / n_pos,
        thresholds=thr,
    )


__all__ = ["ROCCurve", "roc_curve"]
