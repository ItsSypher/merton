"""Atomic firm-level shocks.

These are deterministic multiplicative or additive perturbations to single
:class:`~merton.core.firm.Firm` fields. They are the lego pieces used to
build composite stress scenarios (CCAR severely-adverse, EBA 2023, etc.)
in :mod:`merton.scenarios.predefined`.

Each shock returns a :class:`~merton.scenarios.base.ScenarioResult` so the
chain of transformations is fully introspectable for audit / reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..exceptions import MertonInputError
from .base import Scenario, ScenarioResult

if TYPE_CHECKING:
    from ..core.firm import Firm


@dataclass(slots=True)
class equity_shock(Scenario):  # noqa: N801 - public API uses snake_case for shocks
    """Multiplicative shock to equity value.

    ``factor < 1`` reflects an equity drawdown; ``factor > 1`` a rally.

    Examples
    --------
    >>> from merton import Firm
    >>> from merton.scenarios import equity_shock
    >>> firm = Firm(equity=100, debt_short=10, debt_long=20, equity_vol=0.25)
    >>> stressed = equity_shock(factor=0.7).apply(firm).firm
    >>> stressed.equity
    70.0
    """

    factor: float
    name: str = "equity_shock"

    def __post_init__(self) -> None:
        if self.factor <= 0:
            raise MertonInputError("equity_shock factor must be strictly positive")

    def apply(self, firm: Firm, **kwargs: Any) -> ScenarioResult:
        new_equity = float(np.asarray(firm.equity, dtype=np.float64) * self.factor)
        return ScenarioResult(
            firm=firm.replace(equity=new_equity),
            scenario=self.name,
            parameters={"factor": self.factor},
            description=f"equity × {self.factor:.3f}",
        )


@dataclass(slots=True)
class vol_shock(Scenario):  # noqa: N801
    """Multiplicative shock to equity volatility (``σ_E``)."""

    factor: float
    name: str = "vol_shock"

    def __post_init__(self) -> None:
        if self.factor <= 0:
            raise MertonInputError("vol_shock factor must be strictly positive")

    def apply(self, firm: Firm, **kwargs: Any) -> ScenarioResult:
        if firm.equity_vol is None:
            raise MertonInputError(
                "vol_shock requires firm.equity_vol to be set",
                suggested_fix="Provide equity_vol explicitly on the Firm.",
            )
        new_vol = float(np.asarray(firm.equity_vol, dtype=np.float64) * self.factor)
        return ScenarioResult(
            firm=firm.replace(equity_vol=new_vol),
            scenario=self.name,
            parameters={"factor": self.factor},
            description=f"σ_E × {self.factor:.3f}",
        )


@dataclass(slots=True)
class rate_shock(Scenario):  # noqa: N801
    """Additive shock to the risk-free rate (``r``).

    A 200-bps tightening is ``rate_shock(delta=0.02)``; an easing is
    ``rate_shock(delta=-0.01)``.
    """

    delta: float
    name: str = "rate_shock"

    def apply(self, firm: Firm, **kwargs: Any) -> ScenarioResult:
        new_rf = float(np.asarray(firm.rf, dtype=np.float64) + self.delta)
        return ScenarioResult(
            firm=firm.replace(rf=new_rf),
            scenario=self.name,
            parameters={"delta": self.delta},
            description=f"r + {self.delta * 1e4:+.0f} bps",
        )


@dataclass(slots=True)
class debt_shock(Scenario):  # noqa: N801
    """Multiplicative shock to total debt.

    Applies ``factor`` independently to short- and long-term debt by
    default. Pass ``short_factor`` / ``long_factor`` to differentiate.
    """

    factor: float = 1.0
    short_factor: float | None = None
    long_factor: float | None = None
    name: str = "debt_shock"

    def __post_init__(self) -> None:
        sf = self.short_factor if self.short_factor is not None else self.factor
        lf = self.long_factor if self.long_factor is not None else self.factor
        if sf < 0 or lf < 0:
            raise MertonInputError("debt_shock factors must be non-negative")

    def apply(self, firm: Firm, **kwargs: Any) -> ScenarioResult:
        sf = self.short_factor if self.short_factor is not None else self.factor
        lf = self.long_factor if self.long_factor is not None else self.factor
        new_short = float(np.asarray(firm.debt_short, dtype=np.float64) * sf)
        new_long = float(np.asarray(firm.debt_long, dtype=np.float64) * lf)
        return ScenarioResult(
            firm=firm.replace(debt_short=new_short, debt_long=new_long),
            scenario=self.name,
            parameters={"short_factor": sf, "long_factor": lf},
            description=f"debt_short × {sf:.3f}, debt_long × {lf:.3f}",
        )


__all__ = ["debt_shock", "equity_shock", "rate_shock", "vol_shock"]
