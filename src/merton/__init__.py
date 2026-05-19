"""merton: production-grade Merton structural credit-risk model.

Public API entry points are re-exported here. Heavier submodules (``portfolio``,
``backtest``, ``scenarios``, ``excel``) are imported lazily on first access via
``__getattr__`` so a cold ``import merton`` stays under ~150 ms.

Examples
--------
>>> from merton import Firm, fit
>>> firm = Firm(equity=100, debt_short=20, debt_long=30, equity_vol=0.30)
>>> result = fit(firm)
>>> result.pd >= 0 and result.pd <= 1
True
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - first install before hatch-vcs runs
    __version__ = "0.0.0+unknown"

from . import _config as config
from . import (
    backtest,
    calibration,
    exceptions,
    extensions,
    greeks,
    portfolio,
    reports,
)
from ._backend._numba import warm_cache as _warm_numba_cache
from .batch import batch_fit
from .core.default_point import DefaultPoint
from .core.distance import distance_to_default, prob_of_default
from .core.firm import Firm
from .core.model import MertonModel, fit
from .core.panel import FirmPanel
from .core.physical import physical_pd
from .core.pricing import equity_value
from .core.result import MertonResult
from .core.spread import implied_credit_spread
from .core.term_structure import term_structure_pd

if TYPE_CHECKING:
    from types import ModuleType

# Submodules surfaced lazily through __getattr__ to keep cold-import light.
_LAZY_SUBMODULES = frozenset(
    {
        "diagnostics",
        "excel",
        "io",
        "obs",
        "scenarios",
        "viz",
        "cli",
    }
)


def __getattr__(name: str) -> ModuleType:
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | _LAZY_SUBMODULES)


def warm_cache() -> None:
    """Pre-compile Numba kernels so the first user-facing call is fast.

    Called automatically by the wheel build (via :func:`_warm_numba_cache`)
    and exposed here so users can force the cache to warm explicitly.
    """
    _warm_numba_cache()


__all__ = [
    "DefaultPoint",
    "Firm",
    "FirmPanel",
    "MertonModel",
    "MertonResult",
    "__version__",
    "backtest",
    "batch_fit",
    "calibration",
    "config",
    "distance_to_default",
    "equity_value",
    "excel",
    "exceptions",
    "extensions",
    "fit",
    "greeks",
    "implied_credit_spread",
    "obs",
    "physical_pd",
    "portfolio",
    "prob_of_default",
    "reports",
    "scenarios",
    "term_structure_pd",
    "warm_cache",
]
