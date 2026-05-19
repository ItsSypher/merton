"""Portfolio-level benchmarks: analytic IRB + Monte Carlo."""

from __future__ import annotations

import numpy as np
import pytest

from merton.portfolio import Portfolio, basel_irb_capital, vasicek_var


@pytest.fixture(scope="module")
def pds_5k() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0.001, 0.15, size=5_000)


@pytest.mark.benchmark(group="portfolio-analytic")
def test_basel_irb_capital_vector(benchmark, pds_5k: np.ndarray) -> None:
    benchmark(basel_irb_capital, pds_5k, 0.45, None, 2.5)


@pytest.mark.benchmark(group="portfolio-analytic")
def test_vasicek_var_vector(benchmark, pds_5k: np.ndarray) -> None:
    benchmark(vasicek_var, pds_5k, 0.15, 0.999)


@pytest.mark.benchmark(group="portfolio-mc", min_rounds=1)
def test_simulate_5k_100k_sims(benchmark, pds_5k: np.ndarray) -> None:
    pf = Portfolio(pds_5k, lgd=0.45)
    pf.fit()
    benchmark(pf.simulate, 100_000, seed=42)


@pytest.mark.benchmark(group="portfolio-mc", min_rounds=1)
def test_analytic_vasicek_5k(benchmark, pds_5k: np.ndarray) -> None:
    pf = Portfolio(pds_5k, lgd=0.45)
    pf.fit()
    benchmark(pf.analytic_vasicek, 0.999)
