"""Shared pytest fixtures for the merton test suite."""

from __future__ import annotations

import numpy as np
import pytest

from merton import Firm


@pytest.fixture
def simple_firm() -> Firm:
    """A small, well-behaved single-snapshot firm fixture."""
    return Firm(
        equity=100.0,
        debt_short=20.0,
        debt_long=30.0,
        equity_vol=0.30,
        rf=0.04,
        horizon=1.0,
    )


@pytest.fixture
def aapl_like_firm() -> Firm:
    """A firm with roughly Apple-scale market cap and balance sheet (synthetic)."""
    return Firm(
        equity=3.0e12,
        debt_short=20e9,
        debt_long=90e9,
        equity_vol=0.25,
        rf=0.045,
        dividend_yield=0.006,
        horizon=1.0,
        ticker="AAPL",
    )


@pytest.fixture
def equity_series_firm() -> Firm:
    """A firm with a synthetic equity time series for the VX series-mode test."""
    rng = np.random.default_rng(42)
    n = 252
    # GBM-style returns with realistic vol
    returns = rng.normal(0.10 / n, 0.30 / np.sqrt(n), size=n)
    equity = 100.0 * np.exp(np.cumsum(returns))
    return Firm(
        equity=equity,
        debt_short=20.0,
        debt_long=30.0,
        rf=0.04,
        horizon=1.0,
    )


@pytest.fixture(scope="session")
def numerical_tolerance() -> dict[str, float]:
    return {"rtol": 1e-7, "atol": 1e-10}
