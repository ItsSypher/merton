"""Calibrator-level benchmarks."""

from __future__ import annotations

import numpy as np
import pytest

from merton.calibration import duan_mle, jmr_iterative, vassalou_xing


@pytest.mark.benchmark(group="calibration")
def test_jmr_snapshot(benchmark) -> None:
    benchmark(
        jmr_iterative,
        equity=100.0,
        equity_vol=0.30,
        debt=35.0,
        rf=0.04,
        T=1.0,
    )


@pytest.mark.benchmark(group="calibration")
def test_vassalou_xing_snapshot(benchmark) -> None:
    benchmark(
        vassalou_xing,
        equity=100.0,
        equity_vol=0.30,
        debt=35.0,
        rf=0.04,
        T=1.0,
    )


@pytest.mark.benchmark(group="calibration", min_rounds=3)
def test_vassalou_xing_series(benchmark, equity_series_252: np.ndarray) -> None:
    benchmark(
        vassalou_xing,
        equity=equity_series_252,
        debt=35.0,
        rf=0.04,
        T=1.0,
    )


@pytest.mark.benchmark(group="calibration", min_rounds=2)
def test_duan_mle(benchmark, equity_series_252: np.ndarray) -> None:
    benchmark(
        duan_mle,
        equity_series=equity_series_252,
        debt=35.0,
        rf=0.04,
        T=1.0,
        survivor_bias_correction=False,
    )
