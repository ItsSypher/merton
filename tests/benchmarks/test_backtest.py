"""Backtest-metric benchmarks on large arrays."""

from __future__ import annotations

import numpy as np
import pytest

from merton.backtest import (
    auc,
    brier,
    calibration_curve,
    hosmer_lemeshow,
    ks_statistic,
    roc_curve,
)


@pytest.mark.benchmark(group="backtest")
def test_auc_1m(benchmark, large_pd_default_pairs: tuple[np.ndarray, np.ndarray]) -> None:
    pd, y = large_pd_default_pairs
    benchmark(auc, pd, y)


@pytest.mark.benchmark(group="backtest")
def test_brier_1m(benchmark, large_pd_default_pairs: tuple[np.ndarray, np.ndarray]) -> None:
    pd, y = large_pd_default_pairs
    benchmark(brier, pd, y)


@pytest.mark.benchmark(group="backtest")
def test_ks_1m(benchmark, large_pd_default_pairs: tuple[np.ndarray, np.ndarray]) -> None:
    pd, y = large_pd_default_pairs
    benchmark(ks_statistic, pd, y)


@pytest.mark.benchmark(group="backtest")
def test_roc_curve_1m(benchmark, large_pd_default_pairs: tuple[np.ndarray, np.ndarray]) -> None:
    pd, y = large_pd_default_pairs
    benchmark(roc_curve, pd, y)


@pytest.mark.benchmark(group="backtest")
def test_calibration_curve_1m(
    benchmark, large_pd_default_pairs: tuple[np.ndarray, np.ndarray]
) -> None:
    pd, y = large_pd_default_pairs
    benchmark(calibration_curve, pd, y, bins=20)


@pytest.mark.benchmark(group="backtest")
def test_hosmer_lemeshow_1m(
    benchmark, large_pd_default_pairs: tuple[np.ndarray, np.ndarray]
) -> None:
    pd, y = large_pd_default_pairs
    benchmark(hosmer_lemeshow, pd, y, bins=20)
