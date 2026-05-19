"""Closed-form Greeks of the BSM equity-as-call-option pricing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .._backend._numpy import d1_d2, norm_cdf, norm_pdf
from .._typing import ArrayLike, FloatArray

if TYPE_CHECKING:
    import pandas as pd


def equity_delta(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
    *,
    dividend_yield: ArrayLike = 0.0,
) -> FloatArray:
    """∂E/∂A = e^(-qT)·Φ(d₁) — equity sensitivity to firm asset value."""
    d1, _ = d1_d2(asset_value, asset_vol, debt, rf, T, dividend_yield)
    q = np.asarray(dividend_yield, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    return np.exp(-q * T_) * norm_cdf(d1)


def equity_gamma(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
    *,
    dividend_yield: ArrayLike = 0.0,
) -> FloatArray:
    """∂²E/∂A² = e^(-qT)·φ(d₁) / (A·σ_A·√T)."""
    d1, _ = d1_d2(asset_value, asset_vol, debt, rf, T, dividend_yield)
    A = np.asarray(asset_value, dtype=np.float64)
    s = np.asarray(asset_vol, dtype=np.float64)
    q = np.asarray(dividend_yield, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    return np.exp(-q * T_) * norm_pdf(d1) / (A * s * np.sqrt(T_))


def equity_vega(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
    *,
    dividend_yield: ArrayLike = 0.0,
) -> FloatArray:
    """∂E/∂σ_A = A·e^(-qT)·φ(d₁)·√T."""
    d1, _ = d1_d2(asset_value, asset_vol, debt, rf, T, dividend_yield)
    A = np.asarray(asset_value, dtype=np.float64)
    q = np.asarray(dividend_yield, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    return A * np.exp(-q * T_) * norm_pdf(d1) * np.sqrt(T_)


def equity_theta(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
    *,
    dividend_yield: ArrayLike = 0.0,
) -> FloatArray:
    """∂E/∂T — time decay of equity."""
    d1, d2 = d1_d2(asset_value, asset_vol, debt, rf, T, dividend_yield)
    A = np.asarray(asset_value, dtype=np.float64)
    D = np.asarray(debt, dtype=np.float64)
    s = np.asarray(asset_vol, dtype=np.float64)
    r = np.asarray(rf, dtype=np.float64)
    q = np.asarray(dividend_yield, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    term1 = -A * np.exp(-q * T_) * norm_pdf(d1) * s / (2.0 * np.sqrt(T_))
    term2 = -r * D * np.exp(-r * T_) * norm_cdf(d2)
    term3 = q * A * np.exp(-q * T_) * norm_cdf(d1)
    return term1 + term2 + term3


def equity_rho(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
    *,
    dividend_yield: ArrayLike = 0.0,
) -> FloatArray:
    """∂E/∂r = D·T·e^(-rT)·Φ(d₂)."""
    _, d2 = d1_d2(asset_value, asset_vol, debt, rf, T, dividend_yield)
    D = np.asarray(debt, dtype=np.float64)
    r = np.asarray(rf, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    return D * T_ * np.exp(-r * T_) * norm_cdf(d2)


@dataclass(slots=True, frozen=True)
class GreeksResult:
    """Container for all equity / PD Greeks returned by :func:`greeks`."""

    equity_delta: FloatArray
    equity_gamma: FloatArray
    equity_vega: FloatArray
    equity_theta: FloatArray
    equity_rho: FloatArray
    pd_dleverage: FloatArray
    pd_dvol: FloatArray
    pd_drate: FloatArray

    def to_dict(self) -> dict[str, FloatArray]:
        return {
            "equity_delta": self.equity_delta,
            "equity_gamma": self.equity_gamma,
            "equity_vega": self.equity_vega,
            "equity_theta": self.equity_theta,
            "equity_rho": self.equity_rho,
            "pd_dleverage": self.pd_dleverage,
            "pd_dvol": self.pd_dvol,
            "pd_drate": self.pd_drate,
        }

    def to_pandas(self) -> pd.DataFrame:
        import pandas as pd

        return pd.DataFrame({k: np.atleast_1d(v) for k, v in self.to_dict().items()})


def greeks(
    asset_value: ArrayLike,
    asset_vol: ArrayLike,
    debt: ArrayLike,
    rf: ArrayLike,
    T: ArrayLike,
    *,
    dividend_yield: ArrayLike = 0.0,
) -> GreeksResult:
    """Compute all equity + PD Greeks in one call."""
    from .pd_sensitivity import (
        pd_leverage_sensitivity,
        pd_rate_sensitivity,
        pd_vol_sensitivity,
    )

    return GreeksResult(
        equity_delta=equity_delta(
            asset_value, asset_vol, debt, rf, T, dividend_yield=dividend_yield
        ),
        equity_gamma=equity_gamma(
            asset_value, asset_vol, debt, rf, T, dividend_yield=dividend_yield
        ),
        equity_vega=equity_vega(asset_value, asset_vol, debt, rf, T, dividend_yield=dividend_yield),
        equity_theta=equity_theta(
            asset_value, asset_vol, debt, rf, T, dividend_yield=dividend_yield
        ),
        equity_rho=equity_rho(asset_value, asset_vol, debt, rf, T, dividend_yield=dividend_yield),
        pd_dleverage=pd_leverage_sensitivity(
            asset_value, asset_vol, debt, rf, T, dividend_yield=dividend_yield
        ),
        pd_dvol=pd_vol_sensitivity(
            asset_value, asset_vol, debt, rf, T, dividend_yield=dividend_yield
        ),
        pd_drate=pd_rate_sensitivity(
            asset_value, asset_vol, debt, rf, T, dividend_yield=dividend_yield
        ),
    )


__all__ = [
    "GreeksResult",
    "equity_delta",
    "equity_gamma",
    "equity_rho",
    "equity_theta",
    "equity_vega",
    "greeks",
]
