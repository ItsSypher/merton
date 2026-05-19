"""Tests for the LossDistribution helpers."""

from __future__ import annotations

import numpy as np
import pytest

from merton.exceptions import MertonInputError
from merton.portfolio import LossDistribution


@pytest.fixture
def normal_losses() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.clip(rng.normal(100.0, 30.0, size=10_000), a_min=0.0, a_max=None)


class TestLossDistribution:
    def test_summary_statistics(self, normal_losses) -> None:
        ld = LossDistribution(losses=normal_losses)
        assert ld.mean() == pytest.approx(100.0, abs=2)
        assert ld.std() == pytest.approx(30.0, abs=2)

    def test_var_increases_in_level(self, normal_losses) -> None:
        ld = LossDistribution(losses=normal_losses)
        assert ld.var(0.95) < ld.var(0.99) < ld.var(0.999)

    def test_es_above_var(self, normal_losses) -> None:
        ld = LossDistribution(losses=normal_losses)
        assert ld.expected_shortfall(0.99) >= ld.var(0.99)

    def test_economic_capital_modes(self, normal_losses) -> None:
        ld = LossDistribution(losses=normal_losses)
        ec_el = ld.economic_capital(0.999, capital_basis="el")
        ec_zero = ld.economic_capital(0.999, capital_basis="zero")
        assert ec_zero > ec_el
        with pytest.raises(MertonInputError):
            ld.economic_capital(0.999, capital_basis="bogus")

    def test_rejects_bad_level(self, normal_losses) -> None:
        ld = LossDistribution(losses=normal_losses)
        with pytest.raises(MertonInputError):
            ld.var(1.5)

    def test_rejects_non_1d_losses(self) -> None:
        with pytest.raises(MertonInputError):
            LossDistribution(losses=np.zeros((2, 3)))

    def test_weighted_mean(self) -> None:
        losses = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 8.0])
        ld = LossDistribution(losses=losses, weights=weights)
        assert ld.mean() == pytest.approx(np.average(losses, weights=weights))

    def test_quantile_and_histogram(self, normal_losses) -> None:
        ld = LossDistribution(losses=normal_losses)
        assert isinstance(float(ld.quantile(0.5)), float)
        h, _edges = ld.histogram(bins=20)
        assert h.shape == (20,)

    def test_contributions_require_matrix(self, normal_losses) -> None:
        ld = LossDistribution(losses=normal_losses)
        with pytest.raises(MertonInputError):
            ld.firm_contributions(0.99)

    def test_contributions_modes(self) -> None:
        losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        contribs = np.tile(losses, (3, 1)).T  # shape (5, 3)
        ld = LossDistribution(losses=losses, contributions=contribs)
        es = ld.firm_contributions(level=0.5, method="es")
        var = ld.firm_contributions(level=0.5, method="var")
        assert es.shape == (3,)
        assert var.shape == (3,)
        with pytest.raises(MertonInputError):
            ld.firm_contributions(level=0.5, method="bogus")

    def test_to_pandas(self, normal_losses) -> None:
        ld = LossDistribution(losses=normal_losses)
        df = ld.to_pandas()
        assert len(df) == len(normal_losses)
        assert "loss" in df.columns
