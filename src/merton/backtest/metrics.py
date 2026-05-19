r"""Backtest metrics for PD models.

All metrics here use the Wilcoxon-Mann-Whitney / rank-based identities so
there's no sklearn dependency. They match ``sklearn.metrics`` outputs to
machine precision on the binary-classification case.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .._typing import ArrayLike, FloatArray
from ..exceptions import MertonInputError


def _check_pair(predictions: ArrayLike, defaults: ArrayLike) -> tuple[FloatArray, FloatArray]:
    p = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(defaults, dtype=np.float64)
    if p.shape != y.shape:
        raise MertonInputError(f"predictions shape {p.shape} != defaults shape {y.shape}")
    if p.ndim != 1:
        raise MertonInputError("predictions and defaults must be 1-D")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise MertonInputError("defaults must be 0/1 indicators")
    return p, y


def auc(predictions: ArrayLike, defaults: ArrayLike) -> float:
    r"""Area under the ROC curve.

    Uses the Wilcoxon-Mann-Whitney identity: AUC equals the probability
    that a randomly chosen defaulter has a higher predicted PD than a
    randomly chosen non-defaulter.

    .. math::

        \mathrm{AUC} = \frac{R_+ - n_+(n_+ + 1)/2}{n_+ \cdot n_-}

    where ``R_+`` is the sum of ranks of the positive class, ``n_+`` is
    the number of positives, and ``n_-`` the number of negatives.
    """
    p, y = _check_pair(predictions, defaults)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise MertonInputError("AUC undefined when only one class is present")
    ranks = rankdata(p)
    r_plus = float(ranks[y == 1].sum())
    return (r_plus - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def accuracy_ratio(predictions: ArrayLike, defaults: ArrayLike) -> float:
    """Gini coefficient ``AR = 2 · AUC − 1``."""
    return 2.0 * auc(predictions, defaults) - 1.0


def brier(predictions: ArrayLike, defaults: ArrayLike) -> float:
    """Brier score (mean squared error of probabilistic predictions)."""
    p, y = _check_pair(predictions, defaults)
    return float(np.mean((p - y) ** 2))


def ks_statistic(predictions: ArrayLike, defaults: ArrayLike) -> float:
    """Kolmogorov-Smirnov statistic — max gap between the two empirical CDFs."""
    p, y = _check_pair(predictions, defaults)
    if not (y == 1).any() or not (y == 0).any():
        raise MertonInputError("KS undefined when only one class is present")
    grid = np.sort(np.unique(p))
    pos = np.sort(p[y == 1])
    neg = np.sort(p[y == 0])
    # CDF at each grid point.
    f_pos = np.searchsorted(pos, grid, side="right") / pos.size
    f_neg = np.searchsorted(neg, grid, side="right") / neg.size
    return float(np.max(np.abs(f_pos - f_neg)))


def hosmer_lemeshow(
    predictions: ArrayLike,
    defaults: ArrayLike,
    *,
    bins: int = 10,
) -> tuple[float, float]:
    """Hosmer-Lemeshow goodness-of-fit χ² and (chi², dof).

    Returns ``(chi_squared, degrees_of_freedom)``. P-values come from
    ``scipy.stats.chi2.sf(chi2, dof)``.
    """
    p, y = _check_pair(predictions, defaults)
    n = p.size
    if n < bins * 5:
        raise MertonInputError(f"need at least {bins * 5} obs for Hosmer-Lemeshow with bins={bins}")
    # Equal-count bins by predicted-PD rank.
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    bin_sizes = [n // bins] * bins
    for i in range(n % bins):
        bin_sizes[i] += 1
    chi2 = 0.0
    idx = 0
    for sz in bin_sizes:
        seg_p = p_sorted[idx : idx + sz]
        seg_y = y_sorted[idx : idx + sz]
        observed_pos = float(seg_y.sum())
        expected_pos = float(seg_p.sum())
        observed_neg = sz - observed_pos
        expected_neg = sz - expected_pos
        if expected_pos > 0:
            chi2 += (observed_pos - expected_pos) ** 2 / expected_pos
        if expected_neg > 0:
            chi2 += (observed_neg - expected_neg) ** 2 / expected_neg
        idx += sz
    return float(chi2), float(bins - 2)


__all__ = [
    "accuracy_ratio",
    "auc",
    "brier",
    "hosmer_lemeshow",
    "ks_statistic",
]
