"""LossDistribution container for portfolio Monte Carlo / analytic outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._typing import FloatArray
from ..exceptions import MertonInputError


@dataclass(slots=True)
class LossDistribution:
    """Histogram of simulated portfolio losses (in currency units or fraction).

    Parameters
    ----------
    losses
        1-D array of simulated losses (one per Monte Carlo draw).
    weights
        Optional importance-sampling weights; uniform by default.
    contributions
        Optional ``(n_sims, n_firms)`` matrix of per-firm losses, used for
        marginal-contribution analysis.
    """

    losses: FloatArray
    weights: FloatArray | None = None
    contributions: FloatArray | None = None

    def __post_init__(self) -> None:
        if np.asarray(self.losses).ndim != 1:
            raise MertonInputError("losses must be a 1-D array")

    # --- summary statistics -------------------------------------------------

    def mean(self) -> float:
        if self.weights is None:
            return float(np.mean(self.losses))
        return float(np.average(self.losses, weights=self.weights))

    def std(self) -> float:
        if self.weights is None:
            return float(np.std(self.losses, ddof=1))
        m = self.mean()
        return float(np.sqrt(np.average((self.losses - m) ** 2, weights=self.weights)))

    def var(self, level: float = 0.99) -> float:
        """Quantile-based VaR (``α``-percentile of the loss distribution)."""
        if not 0 < level < 1:
            raise MertonInputError("level must lie in (0, 1)")
        return float(np.quantile(self.losses, level))

    def expected_shortfall(self, level: float = 0.99) -> float:
        """Average loss conditional on exceeding the ``α``-VaR."""
        threshold = self.var(level)
        tail = self.losses[self.losses >= threshold]
        if tail.size == 0:
            return threshold
        return float(np.mean(tail))

    def economic_capital(
        self,
        confidence: float = 0.999,
        *,
        capital_basis: str = "el",
    ) -> float:
        """Capital sufficient to cover unexpected losses at ``confidence``.

        - ``capital_basis="el"`` (default): ``VaR(α) − E[L]`` (Basel-style
          unexpected-loss capital).
        - ``capital_basis="zero"``: ``VaR(α)``; treat the full quantile as
          capital (used by some IFRS-9 implementations).
        """
        v = self.var(confidence)
        if capital_basis == "el":
            return float(v - self.mean())
        if capital_basis == "zero":
            return float(v)
        raise MertonInputError(
            f"unknown capital_basis {capital_basis!r}",
            suggested_fix="Choose 'el' or 'zero'.",
        )

    def quantile(self, q: float | FloatArray) -> float | FloatArray:
        """Empirical quantile of the loss distribution."""
        return np.quantile(self.losses, q)

    def histogram(self, bins: int = 100) -> tuple[FloatArray, FloatArray]:
        return np.histogram(self.losses, bins=bins)

    # --- contribution / decomposition --------------------------------------

    def firm_contributions(
        self,
        level: float = 0.99,
        method: str = "es",
    ) -> FloatArray:
        """Per-firm expected-loss contribution at the given quantile.

        ``method='es'``: mean per-firm loss conditional on portfolio
        losses exceeding the ``α``-VaR. ``method='var'``: per-firm loss at
        the worst Monte Carlo draw that defines the VaR quantile.
        Requires ``contributions`` to be set.
        """
        if self.contributions is None:
            raise MertonInputError(
                "firm_contributions requires the contributions matrix",
                suggested_fix="Pass `record_contributions=True` to Portfolio.simulate.",
            )
        if method == "es":
            threshold = self.var(level)
            mask = self.losses >= threshold
            if mask.sum() == 0:
                return np.zeros(self.contributions.shape[1])
            return self.contributions[mask].mean(axis=0)
        if method == "var":
            idx = int(np.argmin(np.abs(self.losses - self.var(level))))
            return self.contributions[idx]
        raise MertonInputError(
            f"unknown method {method!r}",
            suggested_fix="Choose 'es' or 'var'.",
        )

    def to_pandas(self):  # type: ignore[no-untyped-def]
        import pandas as pd

        return pd.DataFrame({"loss": self.losses})

    # --- repr --------------------------------------------------------------

    def summary(self, level: float = 0.99) -> str:
        return (
            "LossDistribution\n"
            f"  n_samples   : {len(self.losses):,}\n"
            f"  mean (EL)   : {self.mean():,.4f}\n"
            f"  std (UL)    : {self.std():,.4f}\n"
            f"  VaR({level:.3f}) : {self.var(level):,.4f}\n"
            f"  ES ({level:.3f}) : {self.expected_shortfall(level):,.4f}\n"
            f"  EC (99.9%)  : {self.economic_capital(0.999):,.4f}"
        )

    def __repr__(self) -> str:
        return self.summary()


__all__ = ["LossDistribution"]
