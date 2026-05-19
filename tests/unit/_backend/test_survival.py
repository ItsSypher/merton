"""Tests for the first-passage survival-probability helper."""

from __future__ import annotations

import math

import pytest

from merton._backend._survival import (
    log_survival_probability,
    survival_probability,
)


class TestSurvival:
    def test_basic_value(self) -> None:
        # Comfortable solvency: A=130, D=35 → probability should be near 1.
        p = survival_probability(130.0, 35.0, drift=0.10, sigma=0.25, T=1.0)
        assert 0.95 < p <= 1.0

    def test_log_value_matches_log_of_value(self) -> None:
        p = survival_probability(120.0, 50.0, 0.05, 0.30, 1.0)
        log_p = log_survival_probability(120.0, 50.0, 0.05, 0.30, 1.0)
        assert math.isclose(math.log(p), log_p, abs_tol=1e-9)

    def test_underwater_firm_returns_finite_negative(self) -> None:
        # When A <= D the survival probability is zero; we return a large
        # negative log so the optimiser sees a smooth descent direction.
        log_p = log_survival_probability(40.0, 50.0, 0.05, 0.25, 1.0)
        assert log_p == pytest.approx(-1e6)

    def test_higher_vol_lowers_survival(self) -> None:
        p_lo = survival_probability(120.0, 50.0, 0.05, 0.20, 1.0)
        p_hi = survival_probability(120.0, 50.0, 0.05, 0.50, 1.0)
        assert p_lo > p_hi

    def test_longer_horizon_lowers_survival(self) -> None:
        p_short = survival_probability(120.0, 50.0, 0.05, 0.25, 0.5)
        p_long = survival_probability(120.0, 50.0, 0.05, 0.25, 5.0)
        assert p_short > p_long
