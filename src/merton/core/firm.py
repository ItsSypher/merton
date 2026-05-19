"""The :class:`Firm` input container.

A ``Firm`` carries everything a single-firm Merton calibration needs: equity
value, debt structure, risk-free rate, horizon, and the choice of default-point
formula. It is intentionally a tiny frozen dataclass — pydantic is reserved
for the :mod:`merton._config` settings layer, not for hot-path objects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .._typing import ArrayLike, FloatArray
from ..exceptions import MertonInputError
from .default_point import (
    DefaultPoint,
    DefaultPointLike,
    compute_default_point,
)

if TYPE_CHECKING:
    import pandas as pd


@dataclass(slots=True, frozen=True)
class Firm:
    """A single-firm Merton model input.

    Parameters
    ----------
    equity
        Market value of equity (scalar or 1-D time series).
    debt_short
        Short-term (≤ 1y) debt balance.
    debt_long
        Long-term (> 1y) debt balance.
    equity_vol
        Annualised equity volatility ``σ_E`` (decimal). Optional — many
        calibrators estimate it from the equity time series.
    rf
        Risk-free rate ``r``. Defaults to 4 %.
    dividend_yield
        Continuous dividend yield ``q``. Defaults to 0.
    horizon
        Time-to-debt-maturity ``T`` in years. Default 1.0 (1-year DD/PD).
    default_point
        One of ``"kmv"`` (default), ``"total"``, ``"short_only"``, or
        ``"custom"``.
    custom_default_point
        Required when ``default_point == "custom"``.
    ticker, asof
        Optional metadata for traceability in reports / panels.

    Examples
    --------
    >>> firm = Firm(equity=100, debt_short=20, debt_long=30, equity_vol=0.30)
    >>> firm.default_point_value()
    35.0
    >>> firm.total_debt
    50.0
    """

    equity: ArrayLike
    debt_short: ArrayLike
    debt_long: ArrayLike
    equity_vol: ArrayLike | None = None
    rf: ArrayLike = 0.04
    dividend_yield: ArrayLike = 0.0
    horizon: float = 1.0
    default_point: DefaultPointLike = field(default=DefaultPoint.KMV)
    custom_default_point: Callable[[ArrayLike, ArrayLike], FloatArray] | None = None
    ticker: str | None = None
    asof: Any | None = None

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.equity, dtype=np.float64) <= 0):
            raise MertonInputError(
                "equity must be strictly positive",
                suggested_fix="Drop rows where market cap is zero or missing.",
            )
        if np.any(np.asarray(self.debt_short, dtype=np.float64) < 0):
            raise MertonInputError("debt_short must be non-negative")
        if np.any(np.asarray(self.debt_long, dtype=np.float64) < 0):
            raise MertonInputError("debt_long must be non-negative")
        if self.equity_vol is not None and np.any(
            np.asarray(self.equity_vol, dtype=np.float64) <= 0
        ):
            raise MertonInputError("equity_vol must be strictly positive")
        if self.horizon <= 0:
            raise MertonInputError("horizon must be strictly positive")

    # --- derived quantities --------------------------------------------------

    @property
    def total_debt(self) -> float:
        """Sum of short- and long-term debt as a scalar (or array if vectorized)."""
        return float(
            np.asarray(self.debt_short, dtype=np.float64)
            + np.asarray(self.debt_long, dtype=np.float64)
        )

    def default_point_value(self) -> FloatArray:
        """Resolve the default threshold ``L`` per the chosen formula."""
        return compute_default_point(
            self.debt_short,
            self.debt_long,
            kind=self.default_point,
            custom=self.custom_default_point,
        )

    # --- constructors --------------------------------------------------------

    @classmethod
    def from_dict(cls, mapping: dict[str, Any]) -> Firm:
        """Build a :class:`Firm` from a plain dict (e.g. a JSON record)."""
        return cls(**mapping)

    @classmethod
    def from_panel(
        cls,
        df: pd.DataFrame,
        *,
        mapping: dict[str, str] | None = None,
    ) -> Firm:
        """Build a vectorized :class:`Firm` from a pandas DataFrame.

        The DataFrame is expected to have one row per (firm, date) or per
        date, with columns matching the dataclass field names. Use ``mapping``
        to rename source columns to canonical names, e.g.
        ``{"mkt_cap": "equity", "st_debt": "debt_short"}``.
        """
        mapping = mapping or {}
        kw: dict[str, Any] = {}
        for canonical in cls.__dataclass_fields__:
            src = mapping.get(canonical, canonical)
            if src in df.columns:
                kw[canonical] = df[src].to_numpy(dtype=np.float64)
        return cls(**kw)  # type: ignore[arg-type]

    @classmethod
    def from_yfinance(
        cls,
        ticker: str,
        *,
        lookback: str = "2y",
        rf: float = 0.04,
        debt_point: DefaultPointLike = DefaultPoint.KMV,
    ) -> Firm:
        """Quick-start helper: pull equity and balance-sheet data via yfinance.

        Requires the ``[data]`` extra (``pip install "merton[data]"``).
        """
        try:
            import yfinance as yf  # type: ignore[import-not-found]
        except ImportError as err:  # pragma: no cover - exercised in extras tests
            raise ImportError(
                "yfinance is required for Firm.from_yfinance(...). Install merton[data]."
            ) from err
        tkr = yf.Ticker(ticker)
        hist = tkr.history(period=lookback, auto_adjust=False)
        if hist.empty:
            raise MertonInputError(f"No price history returned for ticker {ticker!r}")
        market_cap = float(tkr.fast_info.get("market_cap") or np.nan)
        if not np.isfinite(market_cap):
            raise MertonInputError(
                f"Could not determine market cap for {ticker!r}",
                suggested_fix="Pass equity= explicitly.",
            )
        returns = hist["Close"].pct_change().dropna().to_numpy()
        eq_vol = float(returns.std(ddof=1) * np.sqrt(252))
        balance = tkr.balance_sheet
        try:
            short_debt = float(balance.loc["Current Debt"].iloc[0])
        except (KeyError, IndexError):
            short_debt = 0.0
        try:
            long_debt = float(balance.loc["Long Term Debt"].iloc[0])
        except (KeyError, IndexError):
            long_debt = 0.0
        return cls(
            equity=market_cap,
            debt_short=short_debt,
            debt_long=long_debt,
            equity_vol=eq_vol,
            rf=rf,
            default_point=debt_point,
            ticker=ticker.upper(),
        )

    # --- immutable update ---------------------------------------------------

    def replace(self, **kwargs: Any) -> Firm:  # type: ignore[override]
        """Return a copy with some fields replaced. (Dataclasses are frozen.)"""
        return replace(self, **kwargs)


__all__ = ["Firm"]
