"""Non-vanilla structural credit-risk models.

The package's flagship :class:`merton.MertonModel` covers the Merton (1974)
setup. The extensions below relax one or more of its assumptions:

- :class:`BlackCoxModel` — default can be triggered at any time before
  maturity (first-passage barrier).
- :class:`GeskeModel` — multi-period debt structure via compound options.
- :class:`CreditGradesModel` — random default barrier (RiskMetrics 2002).
- :class:`LelandToftModel` — endogenous default with coupons and taxes.
- :class:`JumpDiffusionModel` — Merton/Zhou (1976/1997) jumps in asset value.
- :class:`LongstaffSchwartzModel` — stochastic Vasicek short rates +
  first-passage default.

Each extension exposes the same ``fit(firm) -> StructuralResult`` contract
so downstream tooling (``viz``, ``backtest``, ``portfolio``) is agnostic
to the model class.
"""

from __future__ import annotations

from .base import StructuralModel, StructuralResult
from .black_cox import BlackCoxModel, black_cox_pd
from .creditgrades import (
    CreditGradesModel,
    creditgrades_pd,
    creditgrades_spread,
    creditgrades_survival,
)
from .geske import GeskeModel, geske_equity_value
from .jump_diffusion import (
    JumpDiffusionModel,
    jump_diffusion_pd,
    simulate_jump_diffusion,
)
from .leland_toft import (
    LelandToftModel,
    leland_toft_debt_value,
    leland_toft_equity_value,
    leland_toft_pd,
    optimal_default_boundary,
)
from .longstaff_schwartz import (
    LongstaffSchwartzModel,
    VasicekParams,
    longstaff_schwartz_pd_analytic,
    longstaff_schwartz_pd_mc,
)

__all__ = [
    "BlackCoxModel",
    "CreditGradesModel",
    "GeskeModel",
    "JumpDiffusionModel",
    "LelandToftModel",
    "LongstaffSchwartzModel",
    "StructuralModel",
    "StructuralResult",
    "VasicekParams",
    "black_cox_pd",
    "creditgrades_pd",
    "creditgrades_spread",
    "creditgrades_survival",
    "geske_equity_value",
    "jump_diffusion_pd",
    "leland_toft_debt_value",
    "leland_toft_equity_value",
    "leland_toft_pd",
    "longstaff_schwartz_pd_analytic",
    "longstaff_schwartz_pd_mc",
    "optimal_default_boundary",
    "simulate_jump_diffusion",
]
