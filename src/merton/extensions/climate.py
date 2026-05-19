r"""ClimateOverlay — climate stress wrapper around any structural model.

The overlay composes a :class:`~merton.scenarios.climate.ClimateScenario`
with a base :class:`~merton.extensions.base.StructuralModel` (Merton,
Black-Cox, CreditGrades, Leland-Toft, etc.). It applies the scenario's
asset writedown to the firm's equity *before* calibration, then scales
the resulting PD by the scenario's sectoral PD multiplier.

Mathematics
-----------
Given a scenario ``s`` and a base structural fit returning
``(A, σ_A, PD_base)``, the overlay produces

.. math::

    PD_{climate}(s) = \min\!\bigl(1,\ m_s \cdot PD(s.\text{apply}(firm))\bigr),

where ``m_s`` is the sectoral PD multiplier and the apply transformation
writes down equity by the scenario's transition + physical writedown.

The DD reported by the overlay is ``-Φ⁻¹(PD_climate)`` for comparability
with the underlying Merton/extension DDs.

Examples
--------
>>> from merton import Firm
>>> from merton.extensions import LelandToftModel
>>> from merton.extensions.climate import ClimateOverlay
>>> from merton.scenarios.climate import Sector
>>> from merton.scenarios.predefined.ngfs import delayed_transition
>>> firm = Firm(equity=100, debt_short=10, debt_long=30, equity_vol=0.25)
>>> base = LelandToftModel()
>>> overlay = ClimateOverlay(base, scenario=delayed_transition(), sector=Sector.ENERGY)
>>> stressed = overlay.fit(firm)
>>> stressed.pd >= base.fit(firm).pd  # climate scenarios should not reduce PD
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import norm

from ..exceptions import MertonInputError
from ..scenarios.climate import ClimateScenario, Sector
from .base import StructuralModel, StructuralResult

if TYPE_CHECKING:
    from ..core.firm import Firm


@dataclass(slots=True)
class ClimateOverlay(StructuralModel):
    r"""Wrap a base structural model with a :class:`ClimateScenario`.

    Parameters
    ----------
    base
        Any :class:`~merton.extensions.base.StructuralModel` (or
        ``MertonModel`` — anything with a ``fit(firm) -> StructuralResult``-
        compatible interface).
    scenario
        Climate scenario to apply.
    sector
        Sector tag for the firm (drives intensity & PD-multiplier lookup).
    intensity
        Override the default sector emission intensity (tCO₂e per $M EV).
    """

    base: Any
    scenario: ClimateScenario
    sector: Sector | str
    intensity: float | None = None
    method: str = field(default="climate_overlay", init=False)

    def __post_init__(self) -> None:
        if not hasattr(self.base, "fit"):
            raise MertonInputError(
                "ClimateOverlay base must expose a fit(firm) method",
                suggested_fix="Pass a MertonModel or any StructuralModel.",
            )
        # Coerce sector to Sector enum at construction so downstream lookups
        # are unambiguous.
        self.sector = Sector(self.sector)

    def fit(self, firm: Firm) -> StructuralResult:
        baseline = self.scenario.apply(firm, sector=self.sector, intensity=self.intensity)
        stressed_firm = baseline.firm
        base_result = self.base.fit(stressed_firm)
        multiplier = self.scenario.pd_multiplier(self.sector)
        stressed_pd = float(np.clip(multiplier * float(base_result.pd), 0.0, 1.0))
        if stressed_pd <= 0.0:
            dd = float("inf")
        elif stressed_pd >= 1.0:
            dd = float("-inf")
        else:
            dd = float(-norm.ppf(stressed_pd))
        diagnostics = {
            "base_method": getattr(base_result, "method", str(type(self.base).__name__)),
            "base_pd": float(base_result.pd),
            "pd_multiplier": multiplier,
            "scenario": self.scenario.name,
            "sector": Sector(self.sector).value,
            "carbon_price": self.scenario.carbon_price(float(firm.horizon)),
            "writedown": baseline.parameters.get("writedown"),
        }
        return StructuralResult(
            firm=firm,
            asset_value=float(base_result.asset_value),
            asset_vol=float(base_result.asset_vol),
            default_point=float(base_result.default_point),
            dd=dd,
            pd=stressed_pd,
            method="climate_overlay",
            diagnostics=diagnostics,
        )


__all__ = ["ClimateOverlay"]
