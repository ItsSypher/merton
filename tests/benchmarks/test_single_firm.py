"""Single-firm benchmarks (pytest-benchmark)."""

from __future__ import annotations

import numpy as np
import pytest

from merton import Firm, MertonModel, distance_to_default, fit, prob_of_default


@pytest.fixture(scope="module")
def single_firm() -> Firm:
    return Firm(
        equity=100.0,
        debt_short=20.0,
        debt_long=30.0,
        equity_vol=0.30,
        rf=0.04,
        horizon=1.0,
    )


@pytest.mark.benchmark(group="single-firm")
def test_distance_to_default_scalar(benchmark) -> None:
    benchmark(distance_to_default, 100.0, 0.25, 60.0, 0.04, 1.0)


@pytest.mark.benchmark(group="single-firm")
def test_prob_of_default_scalar(benchmark) -> None:
    benchmark(prob_of_default, 2.5)


@pytest.mark.benchmark(group="single-firm")
def test_fit_jmr_iterative(benchmark, single_firm: Firm) -> None:
    benchmark(fit, single_firm, method="jmr_iterative")


@pytest.mark.benchmark(group="single-firm")
def test_fit_naive(benchmark, single_firm: Firm) -> None:
    benchmark(fit, single_firm, method="naive")


@pytest.mark.benchmark(group="single-firm")
def test_fit_vassalou_xing(benchmark, single_firm: Firm) -> None:
    benchmark(fit, single_firm, method="vassalou_xing")


@pytest.mark.benchmark(group="single-firm")
def test_model_set_get_params(benchmark) -> None:
    m = MertonModel(method="naive")

    def roundtrip() -> None:
        m.set_params(max_iter=42, tol=1e-8)
        m.get_params()

    benchmark(roundtrip)


@pytest.mark.benchmark(group="vector-ops")
def test_distance_to_default_1m(benchmark) -> None:
    rng = np.random.default_rng(0)
    A = rng.uniform(50.0, 200.0, 1_000_000)
    s = rng.uniform(0.20, 0.40, 1_000_000)
    D = rng.uniform(20.0, 80.0, 1_000_000)
    benchmark(distance_to_default, A, s, D, 0.04, 1.0)
