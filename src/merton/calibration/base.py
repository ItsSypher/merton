"""Calibrator ABC, result dataclass, and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from .._typing import FloatArray
from ..exceptions import MertonInputError

if TYPE_CHECKING:
    from ..core.firm import Firm


@dataclass(slots=True, frozen=True)
class CalibrationResult:
    """Output of any :class:`Calibrator`.

    Attributes
    ----------
    asset_value
        Inferred firm asset value ``A`` (scalar or time series).
    asset_vol
        Inferred annualised asset volatility ``σ_A``.
    asset_drift
        Inferred drift ``μ`` (only set by MLE methods).
    log_likelihood
        Optimised log-likelihood, when applicable.
    covariance
        Asymptotic covariance matrix of parameters, when applicable.
    n_iter
        Iteration count of the solver.
    converged
        Whether the solver hit ``tol`` before ``max_iter``.
    method
        Name of the calibrator that produced this result.
    diagnostics
        Free-form bag of solver-specific telemetry.
    """

    asset_value: float | FloatArray
    asset_vol: float
    asset_drift: float | None = None
    log_likelihood: float | None = None
    covariance: np.ndarray | None = None
    n_iter: int = 0
    converged: bool = True
    method: str = "unknown"
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Calibrator(ABC):
    """Base class for calibration strategies.

    Subclasses override :meth:`fit` and set the class attribute :attr:`method`
    (a short string identifier used for registry lookups).
    """

    method: ClassVar[str] = ""

    def __init_subclass__(cls, /, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.method:
            _REGISTRY[cls.method] = cls

    @abstractmethod
    def fit(self, firm: Firm) -> CalibrationResult:
        """Infer asset value & volatility for ``firm``."""


_REGISTRY: dict[str, type[Calibrator]] = {}


def register(name: str) -> Any:
    """Decorator: register a custom :class:`Calibrator` under ``name``.

    Examples
    --------
    >>> @register("my_method")  # doctest: +SKIP
    ... class MyCalibrator(Calibrator):
    ...     method = "my_method"
    ...
    ...     def fit(self, firm): ...
    """

    def deco(cls: type[Calibrator]) -> type[Calibrator]:
        _REGISTRY[name] = cls
        return cls

    return deco


def get(name: str) -> type[Calibrator]:
    """Look up a calibrator class by name."""
    try:
        return _REGISTRY[name]
    except KeyError as err:
        avail = ", ".join(sorted(_REGISTRY))
        raise MertonInputError(
            f"Unknown calibration method {name!r}",
            suggested_fix=f"Choose one of: {avail}.",
        ) from err


def available_methods() -> list[str]:
    """Return all registered calibration method names."""
    return sorted(_REGISTRY)


__all__ = [
    "CalibrationResult",
    "Calibrator",
    "available_methods",
    "get",
    "register",
]
