"""Exception hierarchy for merton.

Every numerical or operational failure raises a :class:`MertonError` subclass.
Each carries an optional ``suggested_fix`` attribute that gets rendered when
the exception is printed.

Examples
--------
>>> raise CalibrationConvergenceError(
...     "Vassalou-Xing did not converge after 200 steps",
...     suggested_fix="Increase max_iter or relax tol.",
... )
Traceback (most recent call last):
    ...
merton.exceptions.CalibrationConvergenceError: Vassalou-Xing did not converge ...
"""

from __future__ import annotations


class MertonError(Exception):
    """Base class for every error raised by merton."""

    def __init__(self, message: str = "", *, suggested_fix: str | None = None) -> None:
        super().__init__(message)
        self.suggested_fix = suggested_fix

    def __str__(self) -> str:
        base = super().__str__()
        if self.suggested_fix:
            return f"{base}\nSuggested fix: {self.suggested_fix}"
        return base


# --- Input / data quality -----------------------------------------------------


class MertonInputError(MertonError, ValueError):
    """Invalid input values supplied by the caller (NaN, negative debt, etc.)."""


class DimensionMismatchError(MertonInputError):
    """Two or more input arrays have incompatible shapes."""


class NonFiniteInputError(MertonInputError):
    """An input contains NaN or infinity where finite values are required."""


class DataQualityError(MertonError):
    """A passed-in panel of market data is too sparse or contains stale ticks."""


class InsufficientDataError(DataQualityError):
    """Not enough observations to fit / calibrate."""


# --- Calibration --------------------------------------------------------------


class CalibrationError(MertonError):
    """Calibration failed for some reason."""


class CalibrationConvergenceError(CalibrationError):
    """Iterative calibration did not converge inside ``max_iter``."""


# --- Backend dispatch ---------------------------------------------------------


class BackendError(MertonError):
    """Something went wrong with the array-backend dispatch layer."""


class BackendNotAvailableError(BackendError):
    """The requested backend extra is not installed."""


class BackendDispatchError(BackendError):
    """A function could not be dispatched to any available backend."""


# --- Extensions ---------------------------------------------------------------


class ExtensionError(MertonError):
    """An extension model (Black-Cox, Geske, …) raised an error."""


# --- Excel --------------------------------------------------------------------


class ExcelError(MertonError):
    """Something went wrong with the Excel integration path."""


class ManifestInstallError(ExcelError):
    """The Office.js add-in manifest could not be installed."""


class ServerStartupError(ExcelError):
    """The xlwings Server failed to start."""


# --- Warnings -----------------------------------------------------------------


class MertonWarning(UserWarning):
    """Base class for non-fatal warnings emitted by merton."""


class MertonBackendFallbackWarning(MertonWarning):
    """A requested backend was unavailable; we fell back to another."""


class MertonNumericalWarning(MertonWarning):
    """Iterative solver hit a numerical edge case (near-zero σ, etc.)."""


__all__ = [
    "BackendDispatchError",
    "BackendError",
    "BackendNotAvailableError",
    "CalibrationConvergenceError",
    "CalibrationError",
    "DataQualityError",
    "DimensionMismatchError",
    "ExcelError",
    "ExtensionError",
    "InsufficientDataError",
    "ManifestInstallError",
    "MertonBackendFallbackWarning",
    "MertonError",
    "MertonInputError",
    "MertonNumericalWarning",
    "MertonWarning",
    "NonFiniteInputError",
    "ServerStartupError",
]
