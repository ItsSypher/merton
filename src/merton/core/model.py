"""``MertonModel`` — sklearn-style orchestrator over the calibration registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._config import get_config
from ..calibration.base import Calibrator
from ..calibration.base import get as get_calibrator
from .firm import Firm
from .result import MertonResult

if TYPE_CHECKING:
    pass


class MertonModel:
    """Configure and apply a Merton calibration.

    Parameters
    ----------
    method
        Name of a registered calibrator. Default: ``"vassalou_xing"``.
    physical_measure
        If True, ``result.pd`` returns physical PD (requires ``sharpe_ratio``).
    sharpe_ratio
        Asset-class Sharpe ratio used for the physical-measure adjustment.
    tol, max_iter
        Numerical tolerances passed to the underlying calibrator (where the
        calibrator supports them).
    backend
        Force a specific array backend.
    random_state
        Seed for any stochastic component (e.g. bootstrap CIs).

    Examples
    --------
    >>> from merton import Firm, MertonModel
    >>> firm = Firm(equity=100, debt_short=20, debt_long=30, equity_vol=0.30)
    >>> result = MertonModel(method="jmr_iterative").fit(firm)
    >>> result.dd > 0
    True
    """

    def __init__(
        self,
        *,
        method: str | None = None,
        physical_measure: bool = False,
        sharpe_ratio: float | None = None,
        tol: float | None = None,
        max_iter: int | None = None,
        backend: str | None = None,
        random_state: int | None = None,
    ) -> None:
        cfg = get_config()
        self.method = method or cfg.default_calibration_method
        self.physical_measure = physical_measure
        self.sharpe_ratio = sharpe_ratio
        self.tol = tol if tol is not None else cfg.default_tol
        self.max_iter = max_iter if max_iter is not None else cfg.default_max_iter
        self.backend = backend
        self.random_state = random_state

    # ------------------------------------------------------------------

    def fit(self, firm: Firm) -> MertonResult:
        """Calibrate the model for ``firm`` and return a :class:`MertonResult`."""
        calibrator = self._build_calibrator()
        calib = calibrator.fit(firm)
        result = MertonResult.from_calibration(
            firm,
            asset_value=calib.asset_value,
            asset_vol=calib.asset_vol,
            method=self.method,
            asset_drift=calib.asset_drift,
            n_iter=calib.n_iter,
            converged=calib.converged,
            log_likelihood=calib.log_likelihood,
            covariance=calib.covariance,
            diagnostics=calib.diagnostics,
        )
        if self.physical_measure and self.sharpe_ratio is not None:
            physical = result.physical_pd(self.sharpe_ratio)
            return MertonResult(
                firm=result.firm,
                asset_value=result.asset_value,
                asset_vol=result.asset_vol,
                asset_drift=result.asset_drift,
                default_point=result.default_point,
                dd=result.dd,
                pd=physical,
                method=f"{result.method}+physical",
                n_iter=result.n_iter,
                converged=result.converged,
                log_likelihood=result.log_likelihood,
                covariance_=result.covariance_,
                residuals_=result.residuals_,
                diagnostics_={
                    **result.diagnostics_,
                    "sharpe_ratio": self.sharpe_ratio,
                    "rn_pd": result.pd,
                },
            )
        return result

    # ------------------------------------------------------------------
    # sklearn-style helpers
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the configured hyper-parameters."""
        return {
            "method": self.method,
            "physical_measure": self.physical_measure,
            "sharpe_ratio": self.sharpe_ratio,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "backend": self.backend,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> MertonModel:
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid parameter {k!r}")
            setattr(self, k, v)
        return self

    # ------------------------------------------------------------------

    def _build_calibrator(self) -> Calibrator:
        cls = get_calibrator(self.method)
        # Try to pass tol/max_iter where supported.
        try:
            return cls(tol=self.tol, max_iter=self.max_iter)  # type: ignore[call-arg]
        except TypeError:
            return cls()


def fit(firm: Firm, *, method: str | None = None, **kwargs: Any) -> MertonResult:
    """Functional shortcut: ``MertonModel(method=...).fit(firm)``.

    Examples
    --------
    >>> from merton import Firm, fit
    >>> firm = Firm(equity=100, debt_short=20, debt_long=30, equity_vol=0.30)
    >>> result = fit(firm)
    >>> result.dd > 0
    True
    """
    return MertonModel(method=method, **kwargs).fit(firm)


__all__ = ["MertonModel", "fit"]
