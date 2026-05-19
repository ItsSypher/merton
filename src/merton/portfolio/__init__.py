"""Portfolio-level Merton tooling.

- :class:`Portfolio` — container plus Monte Carlo and analytic engines.
- :class:`LossDistribution` — VaR, ES, economic capital, contributions.
- :class:`VasicekFactor` — Basel-IRB single-factor analytics.
- :mod:`merton.portfolio.copulas` — Gaussian and Student-t copulas.
"""

from __future__ import annotations

from .concentration import granularity_adjustment, hhi
from .copulas import GaussianCopula, TCopula
from .correlation import asset_correlation_from_equity, basel_irb_correlation
from .loss_distribution import LossDistribution
from .portfolio import Portfolio, PortfolioResult
from .vasicek_factor import VasicekFactor, basel_irb_capital, vasicek_loss_cdf, vasicek_var

__all__ = [
    "GaussianCopula",
    "LossDistribution",
    "Portfolio",
    "PortfolioResult",
    "TCopula",
    "VasicekFactor",
    "asset_correlation_from_equity",
    "basel_irb_capital",
    "basel_irb_correlation",
    "granularity_adjustment",
    "hhi",
    "vasicek_loss_cdf",
    "vasicek_var",
]
