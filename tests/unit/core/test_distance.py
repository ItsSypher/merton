"""Tests for distance_to_default and prob_of_default."""

from __future__ import annotations

import numpy as np
import pytest

from merton import distance_to_default, prob_of_default
from merton.exceptions import MertonInputError, NonFiniteInputError


class TestDistanceToDefault:
    def test_scalar_inputs_return_scalar(self) -> None:
        dd = distance_to_default(100.0, 0.25, 60.0, 0.04, 1.0)
        assert np.isscalar(dd) or np.ndim(dd) == 0
        assert dd > 0  # well-capitalised firm

    def test_known_value(self) -> None:
        # Hand-computed reference for A=100, σ=0.25, D=60, r=0.04, T=1, q=0:
        # d₁ = (ln(100/60) + (0.04 + 0.25²/2)·1) / (0.25·1) = (0.5108 + 0.07125)/0.25 = 2.3283
        # d₂ = d₁ - σ√T = 2.3283 - 0.25 = 2.0783
        dd = float(distance_to_default(100.0, 0.25, 60.0, 0.04, 1.0))
        assert dd == pytest.approx(2.0783, abs=1e-3)

    def test_vectorized_inputs(self) -> None:
        A = np.array([100.0, 150.0, 80.0])
        dd = distance_to_default(A, 0.25, 60.0, 0.04, 1.0)
        assert dd.shape == (3,)
        # Higher asset value ⇒ larger DD
        assert dd[1] > dd[0] > dd[2]

    def test_higher_vol_lowers_dd(self) -> None:
        dd_low = float(distance_to_default(100.0, 0.20, 60.0, 0.04, 1.0))
        dd_high = float(distance_to_default(100.0, 0.50, 60.0, 0.04, 1.0))
        assert dd_low > dd_high

    def test_rejects_zero_equity(self) -> None:
        with pytest.raises(MertonInputError):
            distance_to_default(0.0, 0.25, 60.0, 0.04, 1.0)

    def test_rejects_zero_vol(self) -> None:
        with pytest.raises(MertonInputError):
            distance_to_default(100.0, 0.0, 60.0, 0.04, 1.0)

    def test_rejects_nan_input(self) -> None:
        with pytest.raises(NonFiniteInputError):
            distance_to_default(float("nan"), 0.25, 60.0, 0.04, 1.0)


class TestProbOfDefault:
    def test_pd_in_unit_interval(self) -> None:
        for dd in [-5.0, -1.0, 0.0, 1.0, 3.0, 10.0]:
            pd = float(prob_of_default(dd))
            assert 0 <= pd <= 1

    def test_pd_decreases_in_dd(self) -> None:
        pds = [float(prob_of_default(dd)) for dd in [0.5, 1.0, 2.0, 3.0]]
        assert all(pds[i] > pds[i + 1] for i in range(len(pds) - 1))

    def test_pd_at_zero_dd(self) -> None:
        assert float(prob_of_default(0.0)) == pytest.approx(0.5, abs=1e-10)

    def test_vectorized(self) -> None:
        pds = prob_of_default(np.array([1.0, 2.0, 3.0]))
        assert pds.shape == (3,)
        assert np.all((pds >= 0) & (pds <= 1))


class TestBackendConsistency:
    """numpy and numba kernels must produce near-identical outputs."""

    def test_numpy_vs_numba(self) -> None:
        from merton._backend import _numba, _numpy

        A = np.array([80.0, 100.0, 120.0, 200.0])
        s = np.array([0.20, 0.25, 0.30, 0.35])
        D = np.array([50.0, 60.0, 70.0, 80.0])
        r = 0.04
        T = 1.0
        q = 0.0
        d2_np = _numpy.distance_to_default_kernel(A, s, D, r, T, q)
        d2_nb = _numba.distance_to_default_kernel(A, s, D, r, T, q)
        np.testing.assert_allclose(d2_np, d2_nb, rtol=1e-7)
