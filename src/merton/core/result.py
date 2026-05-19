"""``MertonResult`` — the frozen container returned by every fit.

The result captures the calibrated state (``asset_value``, ``asset_vol``) plus
derived quantities (DD, PD). Heavy derived outputs (Greeks, term structures,
confidence intervals, summaries) are lazy methods so callers only pay for what
they need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .._typing import ArrayLike, FloatArray
from .distance import distance_to_default, prob_of_default
from .firm import Firm
from .spread import implied_credit_spread

if TYPE_CHECKING:
    from ..greeks.equity import GreeksResult


@dataclass(slots=True, frozen=True, kw_only=True)
class MertonResult:
    """The output of fitting a single firm.

    Attributes
    ----------
    firm
        The input :class:`Firm`.
    asset_value, asset_vol
        Calibrated firm asset value and asset volatility.
    asset_drift
        Calibrated drift (only set by MLE/time-series methods).
    default_point
        Resolved default threshold from the firm's chosen formula.
    dd
        Distance-to-default at ``firm.horizon``.
    pd
        Risk-neutral probability of default at ``firm.horizon``.
    method
        Name of the calibrator used.
    n_iter, converged
        Diagnostics from the solver.
    log_likelihood, covariance_, residuals_
        Statistical output for MLE methods (``None`` for others).
    diagnostics_
        Free-form bag of solver-specific telemetry.
    """

    firm: Firm
    asset_value: float | FloatArray
    asset_vol: float
    asset_drift: float | None = None
    default_point: float | FloatArray
    dd: float | FloatArray
    pd: float | FloatArray
    method: str = "unknown"
    n_iter: int = 0
    converged: bool = True
    log_likelihood: float | None = None
    covariance_: np.ndarray | None = None
    residuals_: np.ndarray | None = None
    diagnostics_: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    @classmethod
    def from_calibration(
        cls,
        firm: Firm,
        asset_value: float | FloatArray,
        asset_vol: float,
        *,
        method: str,
        asset_drift: float | None = None,
        n_iter: int = 0,
        converged: bool = True,
        log_likelihood: float | None = None,
        covariance: np.ndarray | None = None,
        residuals: np.ndarray | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> MertonResult:
        dp = firm.default_point_value()
        dd = distance_to_default(
            asset_value=asset_value,
            asset_vol=asset_vol,
            debt=dp,
            rf=firm.rf,
            T=firm.horizon,
            dividend_yield=firm.dividend_yield,
        )
        pd_ = prob_of_default(dd)
        return cls(
            firm=firm,
            asset_value=asset_value,
            asset_vol=asset_vol,
            asset_drift=asset_drift,
            default_point=dp,
            dd=dd,
            pd=pd_,
            method=method,
            n_iter=n_iter,
            converged=converged,
            log_likelihood=log_likelihood,
            covariance_=covariance,
            residuals_=residuals,
            diagnostics_=diagnostics or {},
        )

    # ------------------------------------------------------------------
    # Lazy / derived quantities
    # ------------------------------------------------------------------

    def pd_term_structure(
        self,
        horizons: ArrayLike = (1 / 12, 3 / 12, 6 / 12, 1.0, 3.0, 5.0),
    ) -> pd.DataFrame:
        """Return a DataFrame of (horizon, DD, PD) across multiple horizons."""
        from .term_structure import term_structure_pd

        # When asset_value is a series, use the last point as the snapshot.
        A = self._scalar_asset_value()
        return term_structure_pd(
            asset_value=A,
            asset_vol=self.asset_vol,
            debt=self._scalar_default_point(),
            rf=float(np.mean(np.asarray(self.firm.rf, dtype=np.float64))),
            horizons=horizons,
            dividend_yield=float(np.mean(np.asarray(self.firm.dividend_yield, dtype=np.float64))),
        )

    def implied_spread(self, lgd: float = 0.6, *, in_bps: bool = True) -> float | FloatArray:
        """Implied credit spread (bps by default) at ``firm.horizon``."""
        return implied_credit_spread(self.pd, T=self.firm.horizon, lgd=lgd, in_bps=in_bps)

    def physical_pd(self, sharpe_ratio: float) -> float | FloatArray:
        """Convert risk-neutral PD to physical PD using the supplied Sharpe ratio."""
        from .physical import physical_pd as _physical_pd

        return _physical_pd(self.pd, self.asset_vol, sharpe_ratio, self.firm.horizon)

    def greeks(self) -> GreeksResult:
        """Compute all closed-form Greeks at the calibrated operating point."""
        from ..greeks import greeks as _greeks

        return _greeks(
            asset_value=self._scalar_asset_value(),
            asset_vol=self.asset_vol,
            debt=self._scalar_default_point(),
            rf=float(np.mean(np.asarray(self.firm.rf, dtype=np.float64))),
            T=self.firm.horizon,
            dividend_yield=float(np.mean(np.asarray(self.firm.dividend_yield, dtype=np.float64))),
        )

    # ------------------------------------------------------------------
    # Exporters
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.firm.ticker,
            "method": self.method,
            "asset_value": self._scalar_asset_value(),
            "asset_vol": self.asset_vol,
            "asset_drift": self.asset_drift,
            "default_point": self._scalar_default_point(),
            "dd": float(
                np.asarray(self.dd).item() if np.ndim(self.dd) == 0 else float(np.mean(self.dd))
            ),
            "pd": float(
                np.asarray(self.pd).item() if np.ndim(self.pd) == 0 else float(np.mean(self.pd))
            ),
            "horizon": self.firm.horizon,
            "converged": self.converged,
            "n_iter": self.n_iter,
        }

    def to_pandas(self) -> pd.DataFrame:
        """Single-row DataFrame summarising the fit."""
        return pd.DataFrame([self.to_dict()])

    def to_polars(self) -> Any:  # pragma: no cover - extras only
        """Return the result as a Polars DataFrame (requires polars installed)."""
        try:
            import polars as pl  # type: ignore[import-not-found]
        except ImportError as err:
            raise ImportError("polars is not installed") from err
        return pl.DataFrame([self.to_dict()])

    def to_excel(self, path: str, *, sheet: str = "Merton") -> None:
        """Write the result to an Excel workbook."""
        df = self.to_pandas()
        with pd.ExcelWriter(path) as writer:
            df.to_excel(writer, sheet_name=sheet, index=False)
            self.pd_term_structure().to_excel(writer, sheet_name="TermStructure", index=False)

    # ------------------------------------------------------------------
    # Summary / repr
    # ------------------------------------------------------------------

    def summary(self) -> str:
        d = self.to_dict()
        ticker = d["ticker"] or "<firm>"
        spread = self.implied_spread()
        spread_val = float(spread if np.ndim(spread) == 0 else np.mean(spread))
        lines = [
            f"MertonResult ({ticker}, method={d['method']})",
            f"  Distance-to-default     : {d['dd']:.4f}",
            f"  Probability of default  : {d['pd']:.6f}",
            f"  Implied spread (LGD=.6) : {spread_val:.2f} bps",
            f"  Asset value             : {d['asset_value']:,.2f}",
            f"  Asset volatility (σ_A)  : {d['asset_vol']:.4f}",
        ]
        if d["asset_drift"] is not None:
            lines.append(f"  Asset drift (μ)         : {d['asset_drift']:.4f}")
        lines.extend(
            [
                f"  Default point           : {d['default_point']:,.2f}",
                f"  Horizon (years)         : {d['horizon']}",
                f"  Solver converged        : {d['converged']} ({d['n_iter']} iters)",
            ]
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def _repr_html_(self) -> str:
        d = self.to_dict()
        rows = [
            ("Method", d["method"]),
            ("Distance-to-default", f"{d['dd']:.4f}"),
            ("Probability of default", f"{d['pd']:.6f}"),
            ("Implied spread (LGD=.6)", f"{float(self.implied_spread()):.2f} bps"),
            ("Asset value", f"{d['asset_value']:,.2f}"),
            ("Asset volatility", f"{d['asset_vol']:.4f}"),
            ("Default point", f"{d['default_point']:,.2f}"),
            ("Horizon (years)", str(d["horizon"])),
            ("Converged", str(d["converged"])),
        ]
        ticker = d["ticker"] or "&lt;firm&gt;"
        body = "".join(
            f"<tr><th style='text-align:left;'>{k}</th><td>{v}</td></tr>" for k, v in rows
        )
        return (
            f"<table><caption><b>MertonResult — {ticker}</b></caption><tbody>{body}</tbody></table>"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scalar_asset_value(self) -> float:
        arr = np.asarray(self.asset_value, dtype=np.float64)
        return float(arr.item() if arr.ndim == 0 else arr[-1])

    def _scalar_default_point(self) -> float:
        arr = np.asarray(self.default_point, dtype=np.float64)
        return float(arr.item() if arr.ndim == 0 else arr.mean())


__all__ = ["MertonResult"]
