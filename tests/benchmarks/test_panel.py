"""Panel benchmarks (1k → 100k firms)."""

from __future__ import annotations

import pandas as pd
import pytest

from merton import batch_fit


@pytest.mark.benchmark(group="panel-1k")
def test_batch_fit_1k_jmr(benchmark, panel_1k: pd.DataFrame) -> None:
    benchmark(batch_fit, panel_1k, method="jmr_iterative", n_jobs=2)


@pytest.mark.benchmark(group="panel-1k")
def test_batch_fit_1k_naive(benchmark, panel_1k: pd.DataFrame) -> None:
    benchmark(batch_fit, panel_1k, method="naive", n_jobs=2)


@pytest.mark.benchmark(group="panel-10k", min_rounds=1)
def test_batch_fit_10k_jmr(benchmark, panel_10k: pd.DataFrame) -> None:
    benchmark(batch_fit, panel_10k, method="jmr_iterative", n_jobs=-1)


@pytest.mark.benchmark(group="panel-10k", min_rounds=1)
def test_batch_fit_10k_naive(benchmark, panel_10k: pd.DataFrame) -> None:
    benchmark(batch_fit, panel_10k, method="naive", n_jobs=-1)


@pytest.mark.slow
@pytest.mark.benchmark(group="panel-100k", min_rounds=1, warmup=False)
def test_batch_fit_100k_naive(benchmark, panel_100k: pd.DataFrame) -> None:
    """100 000-firm naive panel fit. Target <30s on 8 cores."""
    benchmark(batch_fit, panel_100k, method="naive", n_jobs=-1)


@pytest.mark.slow
@pytest.mark.benchmark(group="panel-100k", min_rounds=1, warmup=False)
def test_batch_fit_100k_jmr(benchmark, panel_100k: pd.DataFrame) -> None:
    """100 000-firm JMR iterative panel fit. Target <90s on 8 cores."""
    benchmark(batch_fit, panel_100k, method="jmr_iterative", n_jobs=-1)
