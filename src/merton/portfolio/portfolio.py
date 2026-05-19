"""Portfolio container and Monte Carlo loss simulation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._typing import ArrayLike, FloatArray
from ..exceptions import MertonInputError
from .copulas import GaussianCopula, TCopula
from .loss_distribution import LossDistribution
from .vasicek_factor import basel_irb_correlation, vasicek_var

if TYPE_CHECKING:
    from ..core.firm import Firm


@dataclass(slots=True, frozen=True)
class PortfolioResult:
    pds: FloatArray
    exposures: FloatArray
    lgds: FloatArray
    correlation: FloatArray | None
    expected_loss: float
    diagnostics: dict[str, Any]

    def summary(self) -> str:
        n = len(self.pds)
        return (
            "PortfolioResult\n"
            f"  n_firms         : {n}\n"
            f"  total_exposure  : {self.exposures.sum():,.2f}\n"
            f"  weighted_pd     : {(self.pds * self.exposures).sum() / self.exposures.sum():.6f}\n"
            f"  expected_loss   : {self.expected_loss:,.4f}"
        )


class Portfolio:
    """Container for a portfolio of firms / exposures and the analytic engines.

    Parameters
    ----------
    firms
        Iterable of :class:`merton.Firm` objects, OR a 1-D array of PDs
        when only an aggregate analysis is needed.
    exposures
        Per-firm exposure (currency units). Defaults to ``1.0`` per firm.
    lgd
        Per-firm loss given default. Scalar or array. Defaults to 0.6.
    correlation
        Asset-return correlation matrix or a scalar pairwise ρ. ``None``
        defers to the Basel IRB asset-class correlation derived from each
        firm's PD.
    copula
        ``"gaussian"`` (default) or ``"t"``.
    df
        Degrees of freedom for the t copula.
    asset_class
        Basel asset class for IRB-derived ρ (``"corporate"`` etc.).
    """

    def __init__(
        self,
        firms: Sequence[Firm] | ArrayLike,
        *,
        exposures: ArrayLike | None = None,
        lgd: ArrayLike | float = 0.6,
        correlation: ArrayLike | float | None = None,
        copula: str = "gaussian",
        df: float = 4.0,
        asset_class: str = "corporate",
    ) -> None:
        from ..core.firm import Firm  # local to avoid cycles

        if isinstance(firms, np.ndarray) or (len(firms) > 0 and not isinstance(firms[0], Firm)):
            # User passed an array of PDs directly.
            self.pds = np.asarray(firms, dtype=np.float64)
            self.firms: list[Firm] | None = None
        else:
            self.firms = list(firms)
            # Lazy: PDs are computed at fit() time, not now.
            self.pds = np.full(len(self.firms), np.nan)
        n = len(self.pds)
        if exposures is None:
            self.exposures = np.ones(n, dtype=np.float64)
        else:
            arr = np.asarray(exposures, dtype=np.float64)
            self.exposures = np.broadcast_to(arr, (n,)).astype(np.float64)
        self.lgds = np.broadcast_to(np.asarray(lgd, dtype=np.float64), (n,)).astype(np.float64)
        self.correlation = correlation
        self.copula_name = copula
        self.df = float(df)
        self.asset_class = asset_class

    # ------------------------------------------------------------------

    def fit(self, method: str = "vassalou_xing", **kwargs: Any) -> PortfolioResult:
        """Calibrate Merton on each firm and stash the implied PDs."""
        if self.firms is None:
            # PDs already supplied; just bundle into a PortfolioResult.
            el = float((self.pds * self.lgds * self.exposures).sum())
            return PortfolioResult(
                pds=self.pds,
                exposures=self.exposures,
                lgds=self.lgds,
                correlation=self._effective_correlation(),
                expected_loss=el,
                diagnostics={"calibrated": False},
            )
        from ..core.model import MertonModel

        results = []
        for firm in self.firms:
            try:
                results.append(MertonModel(method=method, **kwargs).fit(firm))
            except Exception:
                results.append(None)
        self.pds = np.array(
            [float(r.pd) if r is not None and r.converged else np.nan for r in results]
        )
        el = float(np.nansum(self.pds * self.lgds * self.exposures))
        return PortfolioResult(
            pds=self.pds,
            exposures=self.exposures,
            lgds=self.lgds,
            correlation=self._effective_correlation(),
            expected_loss=el,
            diagnostics={"calibrated": True, "method": method},
        )

    # ------------------------------------------------------------------

    def simulate(
        self,
        n_sims: int = 100_000,
        *,
        antithetic: bool = True,
        seed: int | None = None,
        record_contributions: bool = False,
    ) -> LossDistribution:
        """Monte Carlo portfolio loss distribution via the chosen copula."""
        if np.any(np.isnan(self.pds)):
            raise MertonInputError(
                "PD vector contains NaN; call Portfolio.fit() before simulate()."
            )
        rng = np.random.default_rng(seed)
        corr = self._effective_correlation()
        if self.copula_name == "gaussian":
            copula = GaussianCopula(corr, n_firms=len(self.pds))
        elif self.copula_name == "t":
            copula = TCopula(corr, n_firms=len(self.pds), df=self.df)
        else:
            raise MertonInputError(
                f"unknown copula {self.copula_name!r}",
                suggested_fix="Choose 'gaussian' or 't'.",
            )
        n_actual = n_sims if not antithetic else n_sims // 2
        u = copula.sample(n_actual, rng=rng)
        if antithetic:
            u = np.concatenate([u, 1.0 - u], axis=0)
        defaults = (u < self.pds[None, :]).astype(np.float64)
        per_firm_loss = defaults * self.lgds[None, :] * self.exposures[None, :]
        losses = per_firm_loss.sum(axis=1)
        return LossDistribution(
            losses=losses,
            weights=None,
            contributions=per_firm_loss if record_contributions else None,
        )

    # ------------------------------------------------------------------

    def analytic_vasicek(
        self,
        confidence: float = 0.999,
        *,
        rho: float | None = None,
    ) -> dict[str, float]:
        """Closed-form Vasicek IRB output (assumes a homogeneous portfolio)."""
        if np.any(np.isnan(self.pds)):
            raise MertonInputError("PD vector contains NaN; call Portfolio.fit() first.")
        # Use exposure-weighted average PD/LGD for the homogeneous approximation.
        w = self.exposures / self.exposures.sum()
        avg_pd = float((self.pds * w).sum())
        avg_lgd = float((self.lgds * w).sum())
        if rho is None:
            rho = float(basel_irb_correlation(avg_pd, asset_class=self.asset_class))
        var = float(vasicek_var(avg_pd, rho, alpha=confidence))
        return {
            "average_pd": avg_pd,
            "average_lgd": avg_lgd,
            "rho": rho,
            "VaR": float(avg_lgd * var * self.exposures.sum()),
            "expected_loss": float(avg_lgd * avg_pd * self.exposures.sum()),
            "unexpected_loss": float(avg_lgd * (var - avg_pd) * self.exposures.sum()),
        }

    def add_firm(self, firm: Firm, exposure: float = 1.0, lgd: float = 0.6) -> Portfolio:
        """Return a new Portfolio with one more firm (immutable update)."""
        if self.firms is None:
            raise MertonInputError("cannot add_firm to a PDs-only Portfolio")
        new_firms = [*self.firms, firm]
        return Portfolio(
            new_firms,
            exposures=np.concatenate([self.exposures, [exposure]]),
            lgd=np.concatenate([self.lgds, [lgd]]),
            correlation=self.correlation,
            copula=self.copula_name,
            df=self.df,
            asset_class=self.asset_class,
        )

    # ------------------------------------------------------------------

    def _effective_correlation(self) -> FloatArray:
        n = len(self.pds)
        if self.correlation is None:
            # Basel-IRB derived per-firm; we form an equicorrelation matrix
            # using the weighted-mean ρ for the copula.
            pd_mean = float(self.pds.mean() if not np.all(np.isnan(self.pds)) else 0.01)
            rho = float(basel_irb_correlation(pd_mean, asset_class=self.asset_class))
            corr = np.full((n, n), rho)
            np.fill_diagonal(corr, 1.0)
            return corr
        if np.isscalar(self.correlation):
            corr = np.full((n, n), float(self.correlation))
            np.fill_diagonal(corr, 1.0)
            return corr
        return np.asarray(self.correlation, dtype=np.float64)


__all__ = ["Portfolio", "PortfolioResult"]
