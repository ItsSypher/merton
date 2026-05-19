"""Structural-model ABC and shared result dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .._typing import FloatArray

if TYPE_CHECKING:
    from ..core.firm import Firm


@dataclass(slots=True, frozen=True)
class StructuralResult:
    """Output of any :class:`StructuralModel`.

    Mirrors the shape of :class:`merton.MertonResult` so downstream tools
    can treat any structural fit uniformly.
    """

    firm: Firm
    asset_value: float | FloatArray
    asset_vol: float
    default_point: float | FloatArray
    dd: float
    pd: float
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        ticker = self.firm.ticker or "<firm>"
        lines = [
            f"StructuralResult ({ticker}, method={self.method})",
            f"  Distance-to-default     : {self.dd:.4f}",
            f"  Probability of default  : {self.pd:.6f}",
            f"  Asset value             : {self.asset_value:,.2f}",
            f"  Asset volatility (σ_A)  : {self.asset_vol:.4f}",
            f"  Default point           : {self.default_point:,.2f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


class StructuralModel(ABC):
    """ABC for structural credit-risk models beyond the flat Merton setup."""

    method: str = ""

    @abstractmethod
    def fit(self, firm: Firm) -> StructuralResult:
        """Return a :class:`StructuralResult` for ``firm``."""


__all__ = ["StructuralModel", "StructuralResult"]
