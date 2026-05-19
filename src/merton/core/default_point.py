"""Default-point formulas.

The *default point* is the liability threshold ``L`` at which the firm is
deemed to default. Several conventions exist:

- **KMV** (``ST + 0.5·LT``): Crosbie & Bohn (2003) empirically-tuned formula.
- **TOTAL** (``ST + LT``): the raw total debt — Merton's original 1974 model.
- **SHORT_ONLY** (``ST``): a stress-test convention.
- **CUSTOM**: a user-supplied callable.

Examples
--------
>>> compute_default_point(20_000_000, 30_000_000, kind="kmv")
35000000.0
>>> compute_default_point(20_000_000, 30_000_000, kind="total")
50000000.0
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import TypeAlias

import numpy as np

from .._typing import ArrayLike, FloatArray
from ..exceptions import MertonInputError


class DefaultPoint(str, Enum):
    """How to compute the default threshold from a firm's balance-sheet items."""

    KMV = "kmv"
    TOTAL = "total"
    SHORT_ONLY = "short_only"
    CUSTOM = "custom"


_KMV_LT_WEIGHT = 0.5

DefaultPointLike: TypeAlias = str | DefaultPoint
DefaultPointCallable: TypeAlias = Callable[[ArrayLike, ArrayLike], FloatArray]


def _coerce(kind: DefaultPointLike) -> DefaultPoint:
    if isinstance(kind, DefaultPoint):
        return kind
    try:
        return DefaultPoint(kind.lower())
    except ValueError as err:
        raise MertonInputError(
            f"Unknown default-point kind {kind!r}",
            suggested_fix="Use one of: 'kmv', 'total', 'short_only', 'custom'.",
        ) from err


def compute_default_point(
    debt_short: ArrayLike,
    debt_long: ArrayLike,
    *,
    kind: DefaultPointLike = DefaultPoint.KMV,
    custom: DefaultPointCallable | None = None,
) -> FloatArray:
    """Return the default threshold given short- and long-term debt.

    Parameters
    ----------
    debt_short, debt_long
        Balance-sheet values (scalar or array). Must be ≥ 0.
    kind
        One of ``"kmv"``, ``"total"``, ``"short_only"``, ``"custom"``.
    custom
        Required when ``kind == "custom"``. Signature
        ``(debt_short, debt_long) -> FloatArray``.
    """
    enum = _coerce(kind)
    st = np.asarray(debt_short, dtype=np.float64)
    lt = np.asarray(debt_long, dtype=np.float64)
    if np.any(st < 0) or np.any(lt < 0):
        raise MertonInputError(
            "debt_short and debt_long must be non-negative",
            suggested_fix="Check the input columns / signs.",
        )
    if enum is DefaultPoint.KMV:
        return st + _KMV_LT_WEIGHT * lt
    if enum is DefaultPoint.TOTAL:
        return st + lt
    if enum is DefaultPoint.SHORT_ONLY:
        return st
    if enum is DefaultPoint.CUSTOM:
        if custom is None:
            raise MertonInputError(
                "kind='custom' requires the `custom` callable",
                suggested_fix="Pass `custom=lambda st, lt: ...`.",
            )
        return np.asarray(custom(st, lt), dtype=np.float64)
    raise MertonInputError(f"Unhandled default-point kind: {enum}")  # pragma: no cover


__all__ = ["DefaultPoint", "DefaultPointLike", "compute_default_point"]
