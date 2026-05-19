"""Asset-correlation helpers."""

from __future__ import annotations

import numpy as np

from .._typing import ArrayLike, FloatArray
from ..exceptions import MertonInputError
from .vasicek_factor import basel_irb_correlation as _basel_corr


def basel_irb_correlation(pd: ArrayLike, *, asset_class: str = "corporate") -> FloatArray:
    """Re-export of :func:`merton.portfolio.vasicek_factor.basel_irb_correlation`."""
    return _basel_corr(pd, asset_class=asset_class)


def asset_correlation_from_equity(
    returns: ArrayLike,
    *,
    leverage: ArrayLike | None = None,
    shrinkage: float = 0.0,
) -> FloatArray:
    """Asset-correlation matrix estimated from a panel of equity log-returns.

    Parameters
    ----------
    returns
        2-D array of shape ``(n_obs, n_firms)`` containing equity log-returns.
    leverage
        Optional 1-D array of length ``n_firms`` of equity/(equity+debt) ratios.
        When supplied, equity correlations are scaled by the leverage product
        to approximate the asset-return correlation matrix.
    shrinkage
        Linear-shrinkage parameter in ``[0, 1]``. ``0`` returns the sample
        correlation; ``1`` returns the identity (no correlation).
    """
    R = np.asarray(returns, dtype=np.float64)
    if R.ndim != 2:
        raise MertonInputError("returns must be a 2-D (n_obs, n_firms) array")
    if not 0 <= shrinkage <= 1:
        raise MertonInputError("shrinkage must lie in [0, 1]")
    centred = R - R.mean(axis=0, keepdims=True)
    cov = np.cov(centred, rowvar=False)
    sd = np.sqrt(np.clip(np.diag(cov), 1e-30, np.inf))
    corr = cov / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    if leverage is not None:
        lev = np.asarray(leverage, dtype=np.float64)
        if lev.ndim != 1 or lev.shape[0] != corr.shape[0]:
            raise MertonInputError("leverage must be a 1-D array matching n_firms")
        corr = corr * np.outer(lev, lev)
        np.fill_diagonal(corr, 1.0)
    if shrinkage > 0:
        corr = (1.0 - shrinkage) * corr + shrinkage * np.eye(corr.shape[0])
    return corr


__all__ = ["asset_correlation_from_equity", "basel_irb_correlation"]
