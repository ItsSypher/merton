"""Excel-facing Python wrappers around the merton math primitives.

Each function below mirrors a public ``=MERTON_*`` formula exposed to Excel
and accepts simple scalar or 1-D array inputs (the natural shape of Excel
cell ranges). When ``xlwings`` is installed, :mod:`merton.excel.server`
applies the ``@func`` / ``@arg`` decorators to the same functions so they
become available as Office.js custom functions.

The wrappers also act as the **single source of truth** for the formula
docstrings and parameter names — the manifest XML and the function-wizard
help text are derived from these signatures.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..calibration import jmr_iterative
from ..core.distance import distance_to_default, prob_of_default
from ..core.spread import implied_credit_spread
from ..core.term_structure import term_structure_pd
from ..extensions import black_cox_pd
from ..greeks import greeks as _greeks
from ..portfolio import VasicekFactor, basel_irb_correlation

# Parameter metadata used by both the xlwings @arg decorators and the
# documentation generators. Keep this list in sync with the function
# signatures below.
ARG_DOCS: dict[str, dict[str, str]] = {
    "merton_dd": {
        "equity": "Market value of equity (currency units).",
        "equity_vol": "Annualised equity volatility (decimal, e.g. 0.25).",
        "debt": "Default threshold — typically short-term debt plus 0.5×long-term debt.",
        "rf": "Risk-free interest rate (annualised decimal).",
        "T": "Horizon in years (e.g. 1 for one-year DD).",
    },
    "merton_pd": {
        "equity": "Market value of equity.",
        "equity_vol": "Annualised equity volatility.",
        "debt": "Default threshold.",
        "rf": "Risk-free rate.",
        "T": "Horizon in years.",
    },
    "merton_spread": {
        "equity": "Market value of equity.",
        "equity_vol": "Annualised equity volatility.",
        "debt": "Default threshold.",
        "rf": "Risk-free rate.",
        "T": "Horizon in years.",
        "lgd": "Loss given default (default 0.6).",
    },
    "merton_asset_value": {
        "equity": "Market value of equity.",
        "equity_vol": "Annualised equity volatility.",
        "debt": "Default threshold.",
        "rf": "Risk-free rate.",
        "T": "Horizon in years.",
    },
    "merton_asset_vol": {
        "equity": "Market value of equity.",
        "equity_vol": "Annualised equity volatility.",
        "debt": "Default threshold.",
        "rf": "Risk-free rate.",
        "T": "Horizon in years.",
    },
    "merton_greeks": {
        "equity": "Market value of equity.",
        "equity_vol": "Annualised equity volatility.",
        "debt": "Default threshold.",
        "rf": "Risk-free rate.",
        "T": "Horizon in years.",
    },
    "merton_pd_term": {
        "equity": "Market value of equity.",
        "equity_vol": "Annualised equity volatility.",
        "debt": "Default threshold.",
        "rf": "Risk-free rate.",
        "horizons": "1-D array of horizons in years (e.g. {0.25, 0.5, 1, 3, 5}).",
    },
    "merton_backtest": {
        "pd_predictions": "Range of predicted PDs.",
        "defaults": "Range of realised 0/1 default indicators (same length).",
        "metric": "Metric name: AUC | Brier | KS | AccuracyRatio.",
    },
    "merton_portfolio_var": {
        "pd": "Probability of default (scalar or weighted-mean across the portfolio).",
        "lgd": "Loss given default (default 0.45 = Basel corporate).",
        "rho": "Asset correlation. Leave blank to use Basel IRB ρ(PD).",
        "alpha": "VaR confidence level (default 0.999).",
        "maturity": "Effective maturity in years (default 1.0).",
    },
    "merton_black_cox": {
        "equity": "Market value of equity.",
        "equity_vol": "Annualised equity volatility.",
        "debt": "Default threshold.",
        "rf": "Risk-free rate.",
        "T": "Horizon in years.",
        "barrier_growth_rate": "Barrier growth rate γ (default 0 = constant barrier).",
    },
}


def _scalarise(x: Any) -> float:
    """Coerce an Excel-style 1×1 range / list / scalar into a plain float."""
    arr = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if arr.size != 1:
        raise ValueError("expected a single number")
    return float(arr.flat[0])


def _invert_asset(
    equity: float,
    equity_vol: float,
    debt: float,
    rf: float,
    T: float,
) -> tuple[float, float]:
    """Solve the JMR two-equation system for (A, σ_A) given the inputs."""
    res = jmr_iterative(equity=equity, equity_vol=equity_vol, debt=debt, rf=rf, T=T)
    return float(res.asset_value), float(res.asset_vol)


# ---------------------------------------------------------------------------
# =MERTON_DD(E, σE, D, r, T)
# ---------------------------------------------------------------------------


def merton_dd(equity: float, equity_vol: float, debt: float, rf: float, T: float) -> float:
    """Distance to default under the Merton (1974) structural model.

    Inverts the equity-as-call-option pricing to recover the asset value
    and asset volatility, then returns ``DD = d₂``.
    """
    e, ev, d, r, t = (_scalarise(x) for x in (equity, equity_vol, debt, rf, T))
    a, sa = _invert_asset(e, ev, d, r, t)
    return float(distance_to_default(a, sa, d, r, t))


# ---------------------------------------------------------------------------
# =MERTON_PD(E, σE, D, r, T)
# ---------------------------------------------------------------------------


def merton_pd(equity: float, equity_vol: float, debt: float, rf: float, T: float) -> float:
    """Risk-neutral probability of default at the horizon ``T``."""
    dd = merton_dd(equity, equity_vol, debt, rf, T)
    return float(prob_of_default(dd))


# ---------------------------------------------------------------------------
# =MERTON_SPREAD(E, σE, D, r, T, LGD)
# ---------------------------------------------------------------------------


def merton_spread(
    equity: float,
    equity_vol: float,
    debt: float,
    rf: float,
    T: float,
    lgd: float = 0.6,
) -> float:
    """Implied credit spread (basis points) given LGD."""
    pd = merton_pd(equity, equity_vol, debt, rf, T)
    return float(implied_credit_spread(pd, _scalarise(T), _scalarise(lgd)))


# ---------------------------------------------------------------------------
# =MERTON_ASSET_VALUE / =MERTON_ASSET_VOL
# ---------------------------------------------------------------------------


def merton_asset_value(equity: float, equity_vol: float, debt: float, rf: float, T: float) -> float:
    """Inferred firm asset value ``A`` (currency units)."""
    e, ev, d, r, t = (_scalarise(x) for x in (equity, equity_vol, debt, rf, T))
    a, _ = _invert_asset(e, ev, d, r, t)
    return a


def merton_asset_vol(equity: float, equity_vol: float, debt: float, rf: float, T: float) -> float:
    """Inferred annualised asset volatility ``σ_A`` (decimal)."""
    e, ev, d, r, t = (_scalarise(x) for x in (equity, equity_vol, debt, rf, T))
    _, sa = _invert_asset(e, ev, d, r, t)
    return sa


# ---------------------------------------------------------------------------
# =MERTON_GREEKS(E, σE, D, r, T)
# ---------------------------------------------------------------------------


def merton_greeks(
    equity: float, equity_vol: float, debt: float, rf: float, T: float
) -> list[list[float]]:
    """Returns a 1×6 array of Greeks: Δ, Γ, Vega, Θ, ρ, ∂PD/∂L.

    The first row is the labels (when the Excel range is 2 rows tall) and
    the second is the values; Excel's dynamic-array engine will spill the
    result automatically.
    """
    e, ev, d, r, t = (_scalarise(x) for x in (equity, equity_vol, debt, rf, T))
    a, sa = _invert_asset(e, ev, d, r, t)
    g = _greeks(asset_value=a, asset_vol=sa, debt=d, rf=r, T=t)
    return [
        ["delta", "gamma", "vega", "theta", "rho", "pd_dleverage"],
        [
            float(g.equity_delta),
            float(g.equity_gamma),
            float(g.equity_vega),
            float(g.equity_theta),
            float(g.equity_rho),
            float(g.pd_dleverage),
        ],
    ]


# ---------------------------------------------------------------------------
# =MERTON_PD_TERM(E, σE, D, r, {horizons…})
# ---------------------------------------------------------------------------


def merton_pd_term(
    equity: float,
    equity_vol: float,
    debt: float,
    rf: float,
    horizons: Sequence[float],
) -> list[list[float]]:
    """PD term structure across horizons. Returns a 2-column dynamic array
    with headers ``[horizon_years, pd]``."""
    e, ev, d, r = (_scalarise(x) for x in (equity, equity_vol, debt, rf))
    horizons_arr = np.atleast_1d(np.asarray(horizons, dtype=np.float64)).ravel()
    # Use T=1 to invert assets (the standard convention) then re-evaluate PDs.
    a, sa = _invert_asset(e, ev, d, r, 1.0)
    df = term_structure_pd(asset_value=a, asset_vol=sa, debt=d, rf=r, horizons=horizons_arr)
    return [["horizon_years", "pd"]] + [
        [float(t), float(p)]
        for t, p in zip(df["horizon_years"].to_numpy(), df["pd"].to_numpy(), strict=True)
    ]


# ---------------------------------------------------------------------------
# =MERTON_BACKTEST(pd_range, defaults_range, "AUC" | "Brier" | "KS" | "AccuracyRatio")
# ---------------------------------------------------------------------------


def merton_backtest(
    pd_predictions: Sequence[float],
    defaults: Sequence[float],
    metric: str = "AUC",
) -> float:
    """Compute a single backtest metric over two equal-length ranges."""
    from ..backtest import accuracy_ratio, auc, brier, ks_statistic

    p = np.asarray(pd_predictions, dtype=np.float64).ravel()
    y = np.asarray(defaults, dtype=np.float64).ravel()
    metric_key = str(metric).strip().lower()
    if metric_key == "auc":
        return float(auc(p, y))
    if metric_key == "brier":
        return float(brier(p, y))
    if metric_key == "ks":
        return float(ks_statistic(p, y))
    if metric_key in {"accuracyratio", "accuracy_ratio", "ar", "gini"}:
        return float(accuracy_ratio(p, y))
    raise ValueError(f"unknown metric {metric!r}; choose AUC, Brier, KS, or AccuracyRatio")


# ---------------------------------------------------------------------------
# =MERTON_PORTFOLIO_VAR(PD, LGD, rho?, alpha?, M?)
# ---------------------------------------------------------------------------


def merton_portfolio_var(
    pd: float,
    lgd: float = 0.45,
    rho: float | None = None,
    alpha: float = 0.999,
    maturity: float = 1.0,
) -> float:
    """Basel-IRB Vasicek single-factor VaR (Loss / Exposure)."""
    pd_v = _scalarise(pd)
    lgd_v = _scalarise(lgd)
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        rho_v = float(basel_irb_correlation(pd_v))
    else:
        rho_v = _scalarise(rho)
    vf = VasicekFactor(pd=pd_v, lgd=lgd_v, rho=rho_v, maturity=_scalarise(maturity))
    return float(vf.lgd * vf.var(alpha=_scalarise(alpha)))


# ---------------------------------------------------------------------------
# =MERTON_BLACK_COX(E, σE, D, r, T, γ?)
# ---------------------------------------------------------------------------


def merton_black_cox(
    equity: float,
    equity_vol: float,
    debt: float,
    rf: float,
    T: float,
    barrier_growth_rate: float = 0.0,
) -> float:
    """Black-Cox first-passage PD with optional barrier-growth rate."""
    e, ev, d, r, t = (_scalarise(x) for x in (equity, equity_vol, debt, rf, T))
    a, sa = _invert_asset(e, ev, d, r, t)
    gamma = _scalarise(barrier_growth_rate)
    return float(black_cox_pd(a, sa, d, r, t, barrier_growth_rate=gamma))


# ---------------------------------------------------------------------------
# Public registry — drives manifest / classic-UDF / server registration.
# ---------------------------------------------------------------------------

EXCEL_FUNCTIONS: tuple[tuple[str, str, Any], ...] = (
    ("MERTON_DD", "Distance to default under Merton (1974).", merton_dd),
    ("MERTON_PD", "Risk-neutral probability of default at the horizon.", merton_pd),
    ("MERTON_SPREAD", "Implied credit spread in basis points.", merton_spread),
    ("MERTON_ASSET_VALUE", "Inferred firm asset value.", merton_asset_value),
    ("MERTON_ASSET_VOL", "Inferred annualised asset volatility.", merton_asset_vol),
    ("MERTON_GREEKS", "Equity Greeks and PD sensitivities (1×6 array).", merton_greeks),
    ("MERTON_PD_TERM", "PD term structure across horizons (2-column array).", merton_pd_term),
    ("MERTON_BACKTEST", "Backtest metric (AUC, Brier, KS, AccuracyRatio).", merton_backtest),
    ("MERTON_PORTFOLIO_VAR", "Basel-IRB Vasicek VaR.", merton_portfolio_var),
    ("MERTON_BLACK_COX", "Black-Cox first-passage PD.", merton_black_cox),
)
"""``(excel_name, description, python_callable)`` tuples driving every consumer
(manifest XML, classic xlwings UDFs, the FastAPI server, and docs)."""
