"""Tests for the Gaussian / Student-t copulas."""

from __future__ import annotations

import numpy as np
import pytest

from merton.exceptions import MertonInputError
from merton.portfolio import GaussianCopula, TCopula


class TestGaussianCopula:
    def test_sample_shape(self) -> None:
        gc = GaussianCopula(0.3, n_firms=5)
        samples = gc.sample(1_000, rng=np.random.default_rng(0))
        assert samples.shape == (1_000, 5)
        # All samples uniform in [0, 1].
        assert np.all((samples >= 0) & (samples <= 1))

    def test_sample_normal_zero_mean(self) -> None:
        gc = GaussianCopula(0.5, n_firms=3)
        z = gc.sample_normal(50_000, rng=np.random.default_rng(0))
        np.testing.assert_allclose(z.mean(axis=0), np.zeros(3), atol=0.05)

    def test_correlation_recovered(self) -> None:
        gc = GaussianCopula(0.5, n_firms=2)
        z = gc.sample_normal(200_000, rng=np.random.default_rng(0))
        corr = np.corrcoef(z, rowvar=False)
        assert corr[0, 1] == pytest.approx(0.5, abs=0.02)

    def test_matrix_input(self) -> None:
        rho_mat = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
        gc = GaussianCopula(rho_mat)
        samples = gc.sample(500, rng=np.random.default_rng(0))
        assert samples.shape == (500, 3)

    def test_invalid_scalar(self) -> None:
        with pytest.raises(MertonInputError):
            GaussianCopula(1.5, n_firms=3)

    def test_scalar_without_n_firms_raises(self) -> None:
        with pytest.raises(MertonInputError):
            GaussianCopula(0.3)

    def test_non_square_raises(self) -> None:
        with pytest.raises(MertonInputError):
            GaussianCopula(np.zeros((3, 2)))


class TestTCopula:
    def test_sample_shape(self) -> None:
        tc = TCopula(0.3, n_firms=4, df=5)
        samples = tc.sample(500, rng=np.random.default_rng(1))
        assert samples.shape == (500, 4)
        assert np.all((samples >= 0) & (samples <= 1))

    def test_rejects_low_df(self) -> None:
        with pytest.raises(MertonInputError):
            TCopula(0.3, n_firms=3, df=2.0)

    def test_tail_dependence_present(self) -> None:
        """t copula's joint upper tail should be heavier than Gaussian."""
        gc = GaussianCopula(0.5, n_firms=2)
        tc = TCopula(0.5, n_firms=2, df=4)
        u_g = gc.sample(50_000, rng=np.random.default_rng(7))
        u_t = tc.sample(50_000, rng=np.random.default_rng(7))
        # P(U1 > 0.99 AND U2 > 0.99) should be larger under t.
        gauss_joint = float(np.mean((u_g[:, 0] > 0.99) & (u_g[:, 1] > 0.99)))
        t_joint = float(np.mean((u_t[:, 0] > 0.99) & (u_t[:, 1] > 0.99)))
        assert t_joint > gauss_joint
