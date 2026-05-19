"""Shared fixtures + RNG seeds for the benchmark suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_panel(n: int, *, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ticker": [f"F{i:08d}" for i in range(n)],
            "equity": rng.uniform(50.0, 500.0, n),
            "debt_short": rng.uniform(10.0, 60.0, n),
            "debt_long": rng.uniform(20.0, 120.0, n),
            "equity_vol": rng.uniform(0.20, 0.45, n),
            "rf": np.full(n, 0.04),
        }
    )


@pytest.fixture(scope="session")
def panel_1k() -> pd.DataFrame:
    return _make_panel(1_000)


@pytest.fixture(scope="session")
def panel_10k() -> pd.DataFrame:
    return _make_panel(10_000)


@pytest.fixture(scope="session")
def panel_100k() -> pd.DataFrame:
    return _make_panel(100_000)


@pytest.fixture(scope="session")
def equity_series_252() -> np.ndarray:
    rng = np.random.default_rng(42)
    log_ret = rng.normal(0.10 / 252, 0.30 / np.sqrt(252), size=252)
    return 100.0 * np.exp(np.cumsum(log_ret))


@pytest.fixture(scope="session")
def large_pd_default_pairs() -> tuple[np.ndarray, np.ndarray]:
    """1 000 000 (pd, default) pairs for backtest-metric benchmarks."""
    rng = np.random.default_rng(0)
    n = 1_000_000
    defaults = rng.binomial(1, 0.05, size=n).astype(np.float64)
    pd = np.clip(0.05 + 0.3 * defaults + rng.normal(0, 0.05, size=n), 0.001, 0.999)
    return pd, defaults
