"""Tests for the Zhou / Merton jump-diffusion model."""

from __future__ import annotations

import numpy as np
import pytest

from merton import Firm, fit
from merton.exceptions import MertonInputError
from merton.extensions import (
    JumpDiffusionModel,
    jump_diffusion_pd,
    simulate_jump_diffusion,
)


class TestJumpDiffusionPD:
    def test_pd_in_unit_interval(self) -> None:
        pd = float(jump_diffusion_pd(100, 0.30, 60, 0.04, 1.0))
        assert 0.0 <= pd <= 1.0

    def test_no_jumps_recovers_merton(self) -> None:
        from merton import distance_to_default, prob_of_default

        pd_jd = float(jump_diffusion_pd(100, 0.30, 60, 0.04, 1.0, jump_intensity=0.0))
        dd = float(distance_to_default(100, 0.30, 60, 0.04, 1.0))
        pd_merton = float(prob_of_default(dd))
        assert pd_jd == pytest.approx(pd_merton, rel=1e-6)

    def test_jumps_increase_pd_for_negative_jump_mean(self) -> None:
        no_jumps = float(jump_diffusion_pd(100, 0.30, 60, 0.04, 1.0, jump_intensity=0.0))
        with_jumps = float(
            jump_diffusion_pd(
                100,
                0.30,
                60,
                0.04,
                1.0,
                jump_intensity=1.0,
                jump_mean=-0.10,
                jump_std=0.20,
            )
        )
        assert with_jumps > no_jumps

    def test_leveraged_higher_pd(self) -> None:
        low = float(jump_diffusion_pd(100, 0.30, 60, 0.04, 1.0))
        hi = float(jump_diffusion_pd(50, 0.30, 60, 0.04, 1.0))
        assert hi > low

    def test_vectorized(self) -> None:
        A = np.array([100.0, 80.0, 50.0])
        out = jump_diffusion_pd(A, 0.3, 60, 0.04, 1.0)
        assert out.shape == (3,)
        assert np.all((out >= 0) & (out <= 1))

    def test_rejects_negative_intensity(self) -> None:
        with pytest.raises(MertonInputError):
            jump_diffusion_pd(100, 0.30, 60, 0.04, 1.0, jump_intensity=-0.1)

    def test_rejects_invalid_jump_std(self) -> None:
        with pytest.raises(MertonInputError):
            jump_diffusion_pd(100, 0.30, 60, 0.04, 1.0, jump_std=0.0)

    def test_rejects_invalid_basic_inputs(self) -> None:
        with pytest.raises(MertonInputError):
            jump_diffusion_pd(0.0, 0.30, 60, 0.04, 1.0)


class TestSimulatePaths:
    def test_paths_shape(self) -> None:
        paths = simulate_jump_diffusion(
            asset_value=100.0,
            drift=0.05,
            sigma=0.30,
            T=1.0,
            n_paths=10,
            n_steps=50,
            jump_intensity=0.5,
            jump_mean=-0.05,
            jump_std=0.15,
            seed=0,
        )
        assert paths.shape == (10, 51)
        # Initial value is the supplied asset_value.
        np.testing.assert_array_equal(paths[:, 0], 100.0)
        # All values remain positive.
        assert np.all(paths > 0)


class TestJumpDiffusionModel:
    def test_fit_smoke(self) -> None:
        firm = Firm(
            equity=100.0, debt_short=20, debt_long=30, equity_vol=0.30, rf=0.04, horizon=1.0
        )
        result = JumpDiffusionModel().fit(firm)
        assert result.method == "jump_diffusion"
        assert 0.0 <= result.pd <= 1.0

    def test_dominates_merton_pd(self) -> None:
        """For negative-mean jumps, JD PD must be ≥ Merton PD."""
        firm = Firm(equity=50.0, debt_short=30, debt_long=50, equity_vol=0.5, rf=0.04, horizon=1.0)
        m = fit(firm, method="jmr_iterative")
        jd = JumpDiffusionModel(jump_intensity=1.0, jump_mean=-0.10, jump_std=0.20).fit(firm)
        assert jd.pd >= m.pd - 1e-10

    def test_requires_equity_vol(self) -> None:
        firm = Firm(equity=100.0, debt_short=20, debt_long=30)
        with pytest.raises(MertonInputError):
            JumpDiffusionModel().fit(firm)
