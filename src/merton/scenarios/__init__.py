"""Scenario and stress-testing primitives.

A *scenario* is a transformation ``Firm → Firm`` (or, more generally,
``Firm → Iterable[Firm]``) that mutates one or more model inputs to
reflect a hypothesised market or balance-sheet state. Composed with
any :class:`~merton.core.model.MertonModel` (or extension), scenarios
produce stressed DD/PD outputs without rewriting the calibration code.

The submodule layout:

- :mod:`merton.scenarios.shocks` — atomic shocks (equity, vol, rate, debt).
- :mod:`merton.scenarios.climate` — climate-transition scenarios driven by
  carbon-price paths and sectoral PD multipliers (TCFD / NGFS-aligned).
- :mod:`merton.scenarios.predefined` — packaged reference scenarios
  (NGFS Phase V net-zero, delayed transition, current policies, fragmented
  world; CCAR / EBA in future phases).

All scenarios implement the same protocol so users can mix and match::

    from merton.scenarios import equity_shock, ClimateScenario
    from merton.scenarios.predefined.ngfs import net_zero_2050

    stress = net_zero_2050()
    stressed_firm = stress.apply(firm, sector=Sector.ENERGY)
"""

from __future__ import annotations

from .base import CompositeScenario, Scenario, ScenarioResult
from .climate import (
    ClimateScenario,
    Sector,
    carbon_price_to_writedown,
    sectoral_carbon_intensity,
)
from .shocks import debt_shock, equity_shock, rate_shock, vol_shock

__all__ = [
    "ClimateScenario",
    "CompositeScenario",
    "Scenario",
    "ScenarioResult",
    "Sector",
    "carbon_price_to_writedown",
    "debt_shock",
    "equity_shock",
    "rate_shock",
    "sectoral_carbon_intensity",
    "vol_shock",
]
