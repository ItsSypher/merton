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

from .._typing import ArrayLike, FloatArray
from .distance import distance_to_default, prob_of_default
from .firm import Firm
from .spread import implied_credit_spread

if TYPE_CHECKING:
    import pandas as pd

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
    dd: float
    pd: float
    dd_series: FloatArray | None = None
    pd_series: FloatArray | None = None
    asset_value_series: FloatArray | None = None
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
        dd_full = distance_to_default(
            asset_value=asset_value,
            asset_vol=asset_vol,
            debt=dp,
            rf=firm.rf,
            T=firm.horizon,
            dividend_yield=firm.dividend_yield,
        )
        pd_full = prob_of_default(dd_full)

        # Scalar headline values use the *latest* observation when we have
        # a time series; otherwise the scalar result of the broadcast.
        if np.ndim(dd_full) == 0:
            dd_scalar = float(dd_full)
            pd_scalar = float(pd_full)
            dd_series_arr: FloatArray | None = None
            pd_series_arr: FloatArray | None = None
        else:
            dd_arr = np.asarray(dd_full, dtype=np.float64)
            pd_arr = np.asarray(pd_full, dtype=np.float64)
            dd_scalar = float(dd_arr[-1])
            pd_scalar = float(pd_arr[-1])
            dd_series_arr = dd_arr
            pd_series_arr = pd_arr

        asset_value_series: FloatArray | None = None
        asset_value_for_field: float | FloatArray
        if np.ndim(asset_value) > 0:
            av_arr = np.asarray(asset_value, dtype=np.float64)
            asset_value_series = av_arr
            asset_value_for_field = float(av_arr[-1])
        else:
            asset_value_for_field = float(np.asarray(asset_value))

        return cls(
            firm=firm,
            asset_value=asset_value_for_field,
            asset_vol=asset_vol,
            asset_drift=asset_drift,
            default_point=dp,
            dd=dd_scalar,
            pd=pd_scalar,
            dd_series=dd_series_arr,
            pd_series=pd_series_arr,
            asset_value_series=asset_value_series,
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

    def confidence_interval(
        self,
        level: float = 0.95,
        method: str = "asymptotic",
        *,
        quantities: tuple[str, ...] = ("asset_vol", "asset_drift", "dd", "pd"),
    ) -> dict[str, Any]:
        """Confidence intervals for fitted parameters and derived quantities.

        Parameters
        ----------
        level
            Confidence level (e.g. 0.95).
        method
            ``"asymptotic"`` uses the MLE observed Fisher information
            (requires :attr:`covariance_` to be set, e.g. from
            ``method='duan_mle'``). ``"bootstrap"`` uses the
            :class:`~merton.calibration.BootstrapResult` stashed in
            ``diagnostics_['bootstrap']`` (requires
            ``MertonModel(n_bootstrap=...)``).
        quantities
            Names to report. Supported: ``"asset_vol"``, ``"asset_drift"``,
            ``"dd"``, ``"pd"``.
        """
        if method == "asymptotic":
            return self._asymptotic_ci(level=level, quantities=quantities)
        if method == "bootstrap":
            return self._bootstrap_ci(level=level, quantities=quantities)
        raise ValueError(f"unknown CI method {method!r}; choose 'asymptotic' or 'bootstrap'")

    def _asymptotic_ci(self, *, level: float, quantities: tuple[str, ...]) -> dict[str, Any]:
        from ..calibration.covariance import delta_method, standard_errors, wald_ci

        if self.covariance_ is None:
            raise ValueError(
                "asymptotic CIs require a fitted Hessian; "
                "use method='duan_mle' or method='bootstrap'."
            )
        cov = np.asarray(self.covariance_, dtype=np.float64)
        se = standard_errors(cov)
        order = self.diagnostics_.get("param_order", ("asset_drift", "asset_vol"))
        idx = {name: i for i, name in enumerate(order)}
        out: dict[str, Any] = {}
        if "asset_vol" in quantities and "asset_vol" in idx:
            out["asset_vol"] = wald_ci(self.asset_vol, float(se[idx["asset_vol"]]), level=level)
        if "asset_drift" in quantities and "asset_drift" in idx and self.asset_drift is not None:
            out["asset_drift"] = wald_ci(
                self.asset_drift, float(se[idx["asset_drift"]]), level=level
            )
        A_scalar = self._scalar_asset_value()
        D_scalar = self._scalar_default_point()
        rf = float(np.mean(np.asarray(self.firm.rf, dtype=np.float64)))
        q = float(np.mean(np.asarray(self.firm.dividend_yield, dtype=np.float64)))
        T = float(self.firm.horizon)

        def _dd_from_params(theta: np.ndarray) -> float:
            sigma = float(theta[idx["asset_vol"]])
            sqrtT = float(np.sqrt(T))
            return float(
                (np.log(A_scalar / D_scalar) + (rf - q - 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
            )

        theta0 = np.zeros(len(order))
        theta0[idx["asset_vol"]] = self.asset_vol
        if "asset_drift" in idx and self.asset_drift is not None:
            theta0[idx["asset_drift"]] = self.asset_drift

        if "dd" in quantities:
            est, se_dd = delta_method(_dd_from_params, theta0, cov)
            out["dd"] = wald_ci(est, se_dd, level=level)
        if "pd" in quantities:
            from scipy.stats import norm as _norm

            def _pd_from_params(theta: np.ndarray) -> float:
                return float(_norm.cdf(-_dd_from_params(theta)))

            est, se_pd = delta_method(_pd_from_params, theta0, cov)
            out["pd"] = wald_ci(est, se_pd, level=level)
        return out

    def _bootstrap_ci(self, *, level: float, quantities: tuple[str, ...]) -> dict[str, Any]:
        bs = self.diagnostics_.get("bootstrap")
        if bs is None:
            raise ValueError(
                "no bootstrap samples on this result; refit with MertonModel(n_bootstrap=...)."
            )
        out: dict[str, Any] = {}
        for name in quantities:
            try:
                out[name] = bs.ci(name, level=level)
            except KeyError:
                continue
        return out

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
            "dd": float(self.dd),
            "pd": float(self.pd),
            "horizon": self.firm.horizon,
            "converged": self.converged,
            "n_iter": self.n_iter,
        }

    def to_pandas(self) -> pd.DataFrame:
        """Single-row DataFrame summarising the fit."""
        import pandas as pd  # lazy: pandas costs ~160 ms to import

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
        import pandas as pd  # lazy

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
