"""Tests for the covariance / Wald-CI / delta-method utilities."""

from __future__ import annotations

import numpy as np
import pytest

from merton.calibration.covariance import (
    ConfInt,
    cov_from_hessian,
    delta_method,
    standard_errors,
    wald_ci,
)
from merton.exceptions import MertonError


class TestCovFromHessian:
    def test_simple_inverse(self) -> None:
        H = np.array([[4.0, 1.0], [1.0, 3.0]])
        cov = cov_from_hessian(H)
        np.testing.assert_allclose(cov, np.linalg.inv(H), rtol=1e-12)

    def test_rejects_non_square(self) -> None:
        with pytest.raises(MertonError):
            cov_from_hessian(np.zeros((2, 3)))

    def test_ridge_on_singular(self) -> None:
        # Singular matrix; ridge should let us still get a result.
        H = np.array([[1.0, 1.0], [1.0, 1.0]])
        cov = cov_from_hessian(H)
        assert cov.shape == (2, 2)


class TestStandardErrors:
    def test_simple(self) -> None:
        cov = np.diag([4.0, 9.0])
        np.testing.assert_allclose(standard_errors(cov), [2.0, 3.0])


class TestWaldCI:
    def test_normal_quantile(self) -> None:
        ci = wald_ci(estimate=0.0, se=1.0, level=0.95)
        # ±1.96
        assert ci.lower == pytest.approx(-1.959964, abs=1e-4)
        assert ci.upper == pytest.approx(1.959964, abs=1e-4)
        assert ci.level == 0.95
        assert ci.method == "asymptotic"

    def test_rejects_bad_level(self) -> None:
        with pytest.raises(MertonError):
            wald_ci(0.0, 1.0, level=1.5)


class TestDeltaMethod:
    def test_identity_function(self) -> None:
        cov = np.array([[1.0, 0.0], [0.0, 4.0]])
        # g(θ) = θ_1 → Var = 1
        est, se = delta_method(lambda t: float(t[0]), np.array([3.0, 7.0]), cov)
        assert est == pytest.approx(3.0)
        assert se == pytest.approx(1.0, abs=1e-4)

    def test_quadratic_function(self) -> None:
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        # g(θ) = θ_1^2 at θ_1 = 2 → ∂g/∂θ_1 = 4 → Var = 16
        est, se = delta_method(lambda t: float(t[0] ** 2), np.array([2.0, 0.0]), cov)
        assert est == pytest.approx(4.0)
        assert se == pytest.approx(4.0, abs=1e-3)


class TestConfInt:
    def test_iter_and_tuple(self) -> None:
        ci = ConfInt(lower=1.0, upper=2.0, level=0.95, method="asymptotic")
        a, b = ci
        assert (a, b) == (1.0, 2.0)
        assert ci.as_tuple() == (1.0, 2.0)
