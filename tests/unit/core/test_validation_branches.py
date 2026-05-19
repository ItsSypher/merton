"""Validation-branch tests for the public core helpers."""

from __future__ import annotations

import numpy as np
import pytest

from merton import (
    distance_to_default,
    implied_credit_spread,
    physical_pd,
    prob_of_default,
)
from merton.core.default_point import DefaultPoint, compute_default_point
from merton.exceptions import MertonInputError


class TestDistanceValidation:
    def test_rejects_negative_debt(self) -> None:
        with pytest.raises(MertonInputError):
            distance_to_default(100.0, 0.25, -10.0, 0.04, 1.0)

    def test_rejects_zero_T(self) -> None:
        with pytest.raises(MertonInputError):
            distance_to_default(100.0, 0.25, 60.0, 0.04, 0.0)

    def test_validate_off_skips_checks(self) -> None:
        # When validate=False the function will run with NaN inputs.
        out = distance_to_default(100.0, 0.25, 60.0, 0.04, 1.0, validate=False)
        assert np.isfinite(float(out))


class TestSpreadValidation:
    def test_negative_T_raises(self) -> None:
        with pytest.raises(MertonInputError):
            implied_credit_spread(0.01, T=-1.0, lgd=0.5)

    def test_pd_out_of_range_raises(self) -> None:
        with pytest.raises(MertonInputError):
            implied_credit_spread(1.5, T=1.0, lgd=0.6)
        with pytest.raises(MertonInputError):
            implied_credit_spread(-0.01, T=1.0, lgd=0.6)

    def test_lgd_out_of_range_raises(self) -> None:
        with pytest.raises(MertonInputError):
            implied_credit_spread(0.01, T=1.0, lgd=1.5)
        with pytest.raises(MertonInputError):
            implied_credit_spread(0.01, T=1.0, lgd=-0.1)

    def test_decimal_output(self) -> None:
        s_dec = float(implied_credit_spread(0.01, 1.0, 0.6, in_bps=False))
        s_bps = float(implied_credit_spread(0.01, 1.0, 0.6, in_bps=True))
        assert s_bps == pytest.approx(s_dec * 10_000.0, rel=1e-9)


class TestPhysicalValidation:
    def test_rejects_non_unit_rn_pd(self) -> None:
        with pytest.raises(MertonInputError):
            physical_pd(1.5, 0.25, 0.5, 1.0)
        with pytest.raises(MertonInputError):
            physical_pd(-0.1, 0.25, 0.5, 1.0)

    def test_rejects_zero_T(self) -> None:
        with pytest.raises(MertonInputError):
            physical_pd(0.01, 0.25, 0.5, 0.0)


class TestDefaultPointEnum:
    def test_unknown_string_raises(self) -> None:
        with pytest.raises(MertonInputError):
            compute_default_point(20.0, 30.0, kind="bogus")

    def test_custom_requires_callable(self) -> None:
        with pytest.raises(MertonInputError):
            compute_default_point(20.0, 30.0, kind="custom")

    def test_negative_debt_rejected(self) -> None:
        with pytest.raises(MertonInputError):
            compute_default_point(-1.0, 30.0)

    def test_enum_passthrough(self) -> None:
        # Passing a DefaultPoint member directly should also work.
        out = compute_default_point(20.0, 30.0, kind=DefaultPoint.TOTAL)
        assert float(out) == 50.0


class TestProbOfDefaultEdgeCases:
    def test_extreme_dd_returns_finite(self) -> None:
        for dd in (-50.0, 50.0):
            v = float(prob_of_default(dd))
            assert np.isfinite(v)
            assert 0.0 <= v <= 1.0
