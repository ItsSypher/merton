"""Tests for solver failure modes (CalibrationConvergenceError, etc.)."""

from __future__ import annotations

import numpy as np
import pytest

from merton import Firm, MertonModel
from merton.calibration import duan_mle, jmr_iterative, naive, vassalou_xing
from merton.calibration._solvers import solve_two_equation
from merton.exceptions import (
    InsufficientDataError,
    MertonInputError,
)


class TestSolveTwoEquation:
    def test_converges_on_typical_inputs(self) -> None:
        a, sa, _, conv = solve_two_equation(E=100.0, sigma_E=0.30, D=50.0, r=0.04, T=1.0)
        assert conv
        assert a > 0 and sa > 0


class TestNaive:
    def test_rejects_zero_equity_vol(self) -> None:
        with pytest.raises(MertonInputError):
            naive(equity=100.0, equity_vol=0.0, debt=35.0, rf=0.04, T=1.0)

    def test_via_model_requires_equity_vol(self) -> None:
        firm = Firm(equity=100, debt_short=20, debt_long=30)
        with pytest.raises(MertonInputError):
            MertonModel(method="naive").fit(firm)


class TestVassalouXingSnapshotErrors:
    def test_scalar_without_equity_vol_raises(self) -> None:
        with pytest.raises(MertonInputError):
            vassalou_xing(equity=100.0, debt=35.0, rf=0.04, T=1.0)

    def test_series_too_short_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            vassalou_xing(equity=np.array([100.0, 101.0, 99.0]), debt=35.0, rf=0.04, T=1.0)


class TestDuanMLEErrors:
    def test_short_series_raises(self) -> None:
        with pytest.raises(InsufficientDataError):
            duan_mle(equity_series=np.array([100.0, 101.0]), debt=35.0, rf=0.04, T=1.0)

    def test_invalid_debt_raises(self) -> None:
        with pytest.raises(MertonInputError):
            duan_mle(
                equity_series=np.linspace(100.0, 110.0, 60),
                debt=0.0,
                rf=0.04,
                T=1.0,
            )


class TestJMRIterativeErrors:
    def test_requires_equity_vol(self) -> None:
        firm = Firm(equity=100, debt_short=20, debt_long=30)
        with pytest.raises(MertonInputError):
            MertonModel(method="jmr_iterative").fit(firm)


class TestUnreachableConvergenceTighten:
    def test_extremely_tight_tol_eventually_succeeds(self) -> None:
        # fsolve hybrid handles tight tols; the path must not raise.
        res = jmr_iterative(
            equity=100.0,
            equity_vol=0.30,
            debt=35.0,
            rf=0.04,
            T=1.0,
            tol=1e-14,
            max_iter=2000,
        )
        assert res.converged


class TestDuanMLECalibratorClass:
    def test_via_model_rejects_scalar_firm(self) -> None:
        firm = Firm(equity=100.0, debt_short=20.0, debt_long=30.0)
        with pytest.raises(InsufficientDataError):
            MertonModel(method="duan_mle").fit(firm)


class TestCovarianceFallbackOnRidge:
    def test_singular_hessian_returns_ridged_inverse(self) -> None:
        # The covariance helper should not raise on a near-singular Hessian.
        from merton.calibration.covariance import cov_from_hessian

        H = np.array([[1.0, 1.0 - 1e-15], [1.0 - 1e-15, 1.0]])
        cov = cov_from_hessian(H)
        assert cov.shape == (2, 2)
