r"""Climate-transition scenarios.

This module provides the building blocks for climate stress testing in the
TCFD / NGFS tradition. A climate scenario specifies:

1. A **carbon-price path** ``c(t)`` (USD per tonne of CO₂-equivalent).
2. **Sectoral asset writedowns** as a function of the carbon price and
   sector-specific emission intensities.
3. **Sectoral PD multipliers** that capture residual default-risk premia
   not absorbed into the asset value (e.g. demand destruction, regulatory
   change, technology obsolescence).

Stress-testing a firm under scenario ``s`` then becomes::

    stressed_firm = s.apply(firm, sector=Sector.ENERGY).firm
    result = MertonModel().fit(stressed_firm)

Sector emission intensities default to the (rounded) IEA 2024
NetZero pathway figures for the listed sectors, expressed as tonnes
CO₂-equivalent per million USD of enterprise value (tCO₂e/M$). Users
can override these with their own bottom-up estimates per portfolio.

References
----------
NGFS (2024). *NGFS Phase V Climate Scenarios for Central Banks and
Supervisors*. https://www.ngfs.net/ngfs-scenarios-portal/
IEA (2024). *World Energy Outlook 2024*. International Energy Agency.
TCFD (2017). *Recommendations of the Task Force on Climate-Related
Financial Disclosures*. Financial Stability Board.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from ..exceptions import MertonInputError
from .base import Scenario, ScenarioResult

if TYPE_CHECKING:
    from ..core.firm import Firm


class Sector(str, Enum):
    """GICS-aligned sector enumeration used throughout the climate module."""

    ENERGY = "energy"
    UTILITIES = "utilities"
    MATERIALS = "materials"
    INDUSTRIALS = "industrials"
    TRANSPORT = "transport"
    REAL_ESTATE = "real_estate"
    CONSUMER = "consumer"
    FINANCIALS = "financials"
    HEALTHCARE = "healthcare"
    TECH = "tech"


# Rounded emission intensities (tCO₂e per $M enterprise value), from
# top-down sector aggregates in IEA WEO 2024 + MSCI Climate Lab Insights
# 2024. These are practitioner defaults — override with company-level
# data when available.
_DEFAULT_INTENSITIES: dict[Sector, float] = {
    Sector.ENERGY: 900.0,
    Sector.UTILITIES: 650.0,
    Sector.MATERIALS: 450.0,
    Sector.INDUSTRIALS: 180.0,
    Sector.TRANSPORT: 320.0,
    Sector.REAL_ESTATE: 90.0,
    Sector.CONSUMER: 60.0,
    Sector.FINANCIALS: 25.0,
    Sector.HEALTHCARE: 40.0,
    Sector.TECH: 30.0,
}


def sectoral_carbon_intensity(sector: Sector | str) -> float:
    """Return the default emission intensity (tCO₂e per $M EV) for a sector.

    Practitioners typically replace these with bottom-up Scope-1+2 figures
    from their internal data; the defaults are reasonable order-of-magnitude
    placeholders for top-down portfolio stress.
    """
    return _DEFAULT_INTENSITIES[Sector(sector)]


def carbon_price_to_writedown(
    carbon_price: float,
    sector: Sector | str,
    *,
    pass_through: float = 0.5,
    intensity: float | None = None,
) -> float:
    r"""Asset writedown factor implied by a carbon price.

    The fraction of enterprise value destroyed by a carbon price ``c`` is

    .. math::

        \text{writedown}(c, \text{sector})
        = \min\!\Bigl(1,\ (1 - \text{pass\_through})
            \cdot \frac{c \cdot I_{\text{sector}}}{10^6}\Bigr),

    where ``I_sector`` is the emission intensity (tCO₂e per $M EV) and
    ``pass_through`` is the share of the cost the firm passes through to
    customers (0 → fully absorbed by equity; 1 → fully passed through).
    The factor returned is the *fractional* writedown (e.g., 0.12 = 12 %
    asset value destroyed); apply it as ``A_stressed = A · (1 − writedown)``.

    Parameters
    ----------
    carbon_price
        Carbon price ``c`` in USD per tonne of CO₂-equivalent.
    sector
        Sector tag (``Sector`` enum or a string the enum can parse).
    pass_through
        Fraction of the carbon cost passed through to customers,
        in ``[0, 1]``. Higher pass-through → smaller writedown.
    intensity
        Override the default sector intensity (tCO₂e per $M EV).
    """
    if carbon_price < 0:
        raise MertonInputError("carbon_price must be non-negative")
    if not 0.0 <= pass_through <= 1.0:
        raise MertonInputError("pass_through must lie in [0, 1]")
    if intensity is None:
        intensity = sectoral_carbon_intensity(sector)
    if intensity < 0:
        raise MertonInputError("intensity must be non-negative")
    raw = (1.0 - pass_through) * carbon_price * intensity / 1.0e6
    return float(min(1.0, max(0.0, raw)))


@dataclass(slots=True)
class ClimateScenario(Scenario):
    r"""A climate-transition stress scenario.

    Parameters
    ----------
    name
        Short identifier (e.g. ``"NGFS Net Zero 2050"``).
    carbon_price_path
        Callable ``t → c(t)`` returning carbon price (USD/tCO₂e) at horizon
        ``t`` (years). Use :func:`carbon_price_curve` for a piecewise-linear
        helper.
    pd_multipliers
        ``{sector: multiplier}`` mapping. ``multiplier > 1`` increases PD
        relative to the no-stress baseline; ``< 1`` decreases it. Sectors
        absent from the mapping default to ``1.0``.
    pass_through
        Fraction of the carbon cost passed through to customers. Default
        0.5 (half passed through, half borne by equity holders).
    physical_writedown
        Additional asset writedown to capture chronic physical risk
        (stranded assets, flooding, etc.) as a fraction of EV per year of
        horizon. Default 0.
    description
        Human-readable summary used in audit trails / reports.
    """

    name: str
    carbon_price_path: Callable[[float], float]
    pd_multipliers: dict[Sector, float] = field(default_factory=dict)
    pass_through: float = 0.5
    physical_writedown: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.pass_through <= 1.0:
            raise MertonInputError("pass_through must lie in [0, 1]")
        if self.physical_writedown < 0.0:
            raise MertonInputError("physical_writedown must be non-negative")
        # Normalise keys: accept either strings or Sector members.
        if any(not isinstance(k, Sector) for k in self.pd_multipliers):
            self.pd_multipliers = {Sector(k): float(v) for k, v in self.pd_multipliers.items()}
        for k, v in self.pd_multipliers.items():
            if v < 0:
                raise MertonInputError(f"pd_multipliers[{k}] must be non-negative")

    def carbon_price(self, horizon: float) -> float:
        """Return the carbon price at ``horizon`` (years from now)."""
        if horizon < 0:
            raise MertonInputError("horizon must be non-negative")
        return float(self.carbon_price_path(horizon))

    def asset_writedown(
        self,
        horizon: float,
        sector: Sector | str,
        *,
        intensity: float | None = None,
    ) -> float:
        """Total fractional asset writedown (transition + physical)."""
        transition = carbon_price_to_writedown(
            self.carbon_price(horizon),
            sector,
            pass_through=self.pass_through,
            intensity=intensity,
        )
        physical = min(1.0, self.physical_writedown * horizon)
        return float(min(1.0, transition + physical * (1.0 - transition)))

    def pd_multiplier(self, sector: Sector | str) -> float:
        """Return the PD multiplier for ``sector`` (1.0 if unset)."""
        return float(self.pd_multipliers.get(Sector(sector), 1.0))

    def apply(
        self,
        firm: Firm,
        *,
        sector: Sector | str | None = None,
        intensity: float | None = None,
        **kwargs: Any,
    ) -> ScenarioResult:
        """Apply the climate stress to ``firm`` and return a stressed copy.

        The transformation only writes down equity (the observable input
        to Merton). ``ClimateOverlay`` is the recommended wrapper if you
        also want PD-multiplier scaling applied to the structural output.
        """
        if sector is None:
            raise MertonInputError(
                "ClimateScenario.apply requires sector=...",
                suggested_fix="Pass sector=Sector.ENERGY (or the appropriate enum).",
            )
        horizon = float(firm.horizon)
        writedown = self.asset_writedown(horizon, sector, intensity=intensity)
        new_equity = float(np.asarray(firm.equity, dtype=np.float64) * (1.0 - writedown))
        if new_equity <= 0:
            # Equity wiped out → bump to a tiny positive value so downstream
            # math stays defined; PD will be ~1.
            new_equity = max(new_equity, 1e-6)
        return ScenarioResult(
            firm=firm.replace(equity=new_equity),
            scenario=self.name,
            parameters={
                "sector": str(Sector(sector)),
                "horizon": horizon,
                "carbon_price": self.carbon_price(horizon),
                "writedown": writedown,
                "pd_multiplier": self.pd_multiplier(sector),
            },
            description=self.description
            or f"{self.name}: equity × {1.0 - writedown:.3f} for {Sector(sector).value}",
        )


def carbon_price_curve(
    knots: list[tuple[float, float]],
) -> Callable[[float], float]:
    """Build a piecewise-linear carbon-price curve from ``(t, price)`` knots.

    Knots are sorted by time; beyond the last knot the function holds the
    final value constant (the NGFS convention).

    Examples
    --------
    >>> from merton.scenarios.climate import carbon_price_curve
    >>> path = carbon_price_curve([(0.0, 50.0), (10.0, 300.0)])
    >>> path(5.0)
    175.0
    """
    if not knots:
        raise MertonInputError("carbon_price_curve requires at least one knot")
    sorted_knots = sorted(knots, key=lambda kv: kv[0])
    ts = np.array([t for t, _ in sorted_knots], dtype=np.float64)
    ps = np.array([p for _, p in sorted_knots], dtype=np.float64)
    if np.any(ps < 0):
        raise MertonInputError("carbon prices must be non-negative")

    def path(t: float) -> float:
        return float(np.interp(t, ts, ps, left=ps[0], right=ps[-1]))

    return path


__all__ = [
    "ClimateScenario",
    "Sector",
    "carbon_price_curve",
    "carbon_price_to_writedown",
    "sectoral_carbon_intensity",
]
