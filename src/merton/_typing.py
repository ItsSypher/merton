"""Shared type aliases for the merton package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd

FloatArray: TypeAlias = NDArray[np.float64]
"""1-D or N-D array of float64 values."""

IntArray: TypeAlias = NDArray[np.int64]

BoolArray: TypeAlias = NDArray[np.bool_]

Scalar: TypeAlias = float | int

ArrayLike: TypeAlias = Union[Scalar, FloatArray, "pd.Series[float]", list[float]]
"""Scalars, NumPy arrays, pandas Series, or plain lists are all accepted."""

NumberLike: TypeAlias = float | int | FloatArray


class SupportsArray(Protocol):
    """Anything implementing the ``__array__`` protocol."""

    def __array__(self, dtype: Any = ..., copy: bool | None = ...) -> Any: ...
