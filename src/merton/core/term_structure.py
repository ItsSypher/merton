"""PD term-structure generation.

Given a fitted Merton model (asset value, asset volatility, default point,
risk-free rate), return PDs across an array of horizons in one call. The
helper here is exposed via :meth:`MertonResult.pd_term_structure` for
convenience.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from .._typing import ArrayLike
from .distance import distance_to_default, prob_of_default

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_HORIZONS: tuple[float, ...] = (1 / 12, 3 / 12, 6 / 12, 1.0, 3.0, 5.0)


def term_structure_pd(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    *,
    horizons: Iterable[float] = DEFAULT_HORIZONS,
    dividend_yield: ArrayLike = 0.0,
) -> pd.DataFrame:
    """Compute PDs for several horizons in one shot.

    Returns
    -------
    pandas.DataFrame
        Columns: ``horizon_years`` (the input grid), ``dd`` (distance to
        default at that horizon), ``pd`` (risk-neutral PD).
    """
    import pandas as pd  # lazy: pandas costs ~160 ms to import

    horizon_arr = np.asarray(list(horizons), dtype=np.float64)
    dd = distance_to_default(
        asset_value, asset_vol, debt, rf, horizon_arr, dividend_yield=dividend_yield
    )
    pd_vals = prob_of_default(dd)
    return pd.DataFrame(
        {
            "horizon_years": horizon_arr,
            "dd": np.asarray(dd, dtype=np.float64),
            "pd": np.asarray(pd_vals, dtype=np.float64),
        }
    )


__all__ = ["DEFAULT_HORIZONS", "term_structure_pd"]
