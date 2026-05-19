"""Non-vanilla structural credit-risk models.

The package's flagship :class:`merton.MertonModel` covers the Merton (1974)
setup. The extensions below relax one or more of its assumptions:

- :class:`BlackCoxModel` — default can be triggered at any time before
  maturity (first-passage barrier).
- :class:`GeskeModel` — multi-period debt structure via compound options.

Each extension exposes the same ``fit(firm) -> StructuralResult`` contract
so downstream tooling (``viz``, ``backtest``, ``portfolio``) is agnostic
to the model class.
"""

from __future__ import annotations

from .base import StructuralModel, StructuralResult
from .black_cox import BlackCoxModel, black_cox_pd
from .geske import GeskeModel, geske_equity_value

__all__ = [
    "BlackCoxModel",
    "GeskeModel",
    "StructuralModel",
    "StructuralResult",
    "black_cox_pd",
    "geske_equity_value",
]
