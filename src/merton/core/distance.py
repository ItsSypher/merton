"""Distance-to-default and probability-of-default primitives.

These are the most-called functions in the whole package. They accept any
combination of scalar and array inputs (NumPy-style broadcasting) and
dispatch to the fastest available backend.

Examples
--------
>>> from merton import distance_to_default, prob_of_default
>>> dd = distance_to_default(100, 0.25, 60, 0.04, 1.0)
>>> float(round(dd, 4))
2.0783
>>> float(round(prob_of_default(dd), 6))
0.018841
"""

from __future__ import annotations

import numpy as np

from .._backend import get_kernel, resolve
from .._typing import ArrayLike, FloatArray
from ..exceptions import MertonInputError, NonFiniteInputError


def _size(*arrays: object) -> int:
    sz = 1
    for a in arrays:
        sz = max(sz, int(np.size(np.asarray(a))))
    return sz


def _validate(
    asset_value: object,
    asset_vol: object,
    debt: object,
    rf: object,
    T: object,
) -> None:
    asset_value_arr = np.asarray(asset_value, dtype=np.float64)
    asset_vol_arr = np.asarray(asset_vol, dtype=np.float64)
    debt_arr = np.asarray(debt, dtype=np.float64)
    T_arr = np.asarray(T, dtype=np.float64)
    rf_arr = np.asarray(rf, dtype=np.float64)
    for name, arr in (
        ("asset_value", asset_value_arr),
        ("asset_vol", asset_vol_arr),
        ("debt", debt_arr),
        ("rf", rf_arr),
        ("T", T_arr),
    ):
        if not np.all(np.isfinite(arr)):
            raise NonFiniteInputError(
                f"{name} contains NaN or infinity",
                suggested_fix="Drop or impute the offending rows before calling.",
            )
    if np.any(asset_value_arr <= 0):
        raise MertonInputError("asset_value must be strictly positive")
    if np.any(asset_vol_arr <= 0):
        raise MertonInputError("asset_vol must be strictly positive")
    if np.any(debt_arr <= 0):
        raise MertonInputError("debt must be strictly positive")
    if np.any(T_arr <= 0):
        raise MertonInputError("T must be strictly positive")


def distance_to_default(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
    *,
    dividend_yield: ArrayLike = 0.0,
    backend: str | None = None,
    validate: bool = True,
) -> FloatArray:
    r"""Compute Merton's distance-to-default ``DD = d_2``.

    .. math::

        DD = \frac{\ln(A/D) + (r - q - \sigma_A^2 / 2)\, T}{\sigma_A \sqrt{T}}

    Parameters
    ----------
    asset_value
        Firm asset value(s) ``A`` (strictly positive).
    asset_vol
        Annualised asset volatility ``σ_A`` (strictly positive).
    debt
        Default threshold ``D`` (strictly positive). For real firms this is
        typically computed via :func:`merton.core.compute_default_point`.
    rf
        Risk-free rate ``r`` (decimal).
    T
        Horizon in years (strictly positive).
    dividend_yield
        Continuous dividend yield ``q``. Defaults to 0.
    backend
        Force a particular backend.
    validate
        If True, check for non-finite and out-of-domain inputs.

    Returns
    -------
    FloatArray
        ``DD`` — number of asset-volatility standard deviations between the
        log of assets and the log of the default threshold.

    References
    ----------
    Merton (1974). Vassalou & Xing (2004). Crosbie & Bohn (2003).

    Examples
    --------
    >>> import numpy as np
    >>> dd = distance_to_default(
    ...     asset_value=np.array([100.0, 150.0]),
    ...     asset_vol=0.25,
    ...     debt=60.0,
    ...     rf=0.04,
    ...     T=1.0,
    ... )
    >>> dd.shape
    (2,)
    """
    if validate:
        _validate(asset_value, asset_vol, debt, rf, T)
    chosen = resolve(
        asset_value,
        asset_vol,
        debt,
        rf,
        T,
        backend=backend,
        size=_size(asset_value, asset_vol, debt, rf, T),
    )
    kernel = get_kernel(chosen, "distance_to_default_kernel")
    return kernel(asset_value, asset_vol, debt, rf, T, dividend_yield)


def prob_of_default(
    dd: ArrayLike,
    *,
    backend: str | None = None,
) -> FloatArray:
    r"""Risk-neutral probability of default given the distance-to-default.

    .. math::

        \text{PD} = \Phi(-DD)

    where :math:`\Phi` is the standard-normal CDF.

    Examples
    --------
    >>> import numpy as np
    >>> float(round(prob_of_default(2.5), 5))
    0.00621
    """
    chosen = resolve(dd, backend=backend, size=_size(dd))
    kernel = get_kernel(chosen, "prob_of_default_kernel")
    return kernel(dd)


__all__ = ["distance_to_default", "prob_of_default"]
