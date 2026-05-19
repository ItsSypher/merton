"""Tests for the climate scenario module."""

from __future__ import annotations

import pytest

from merton import Firm
from merton.exceptions import MertonInputError
from merton.scenarios.climate import (
    ClimateScenario,
    Sector,
    carbon_price_curve,
    carbon_price_to_writedown,
    sectoral_carbon_intensity,
)


class TestSector:
    def test_all_sectors_have_intensities(self) -> None:
        for sec in Sector:
            assert sectoral_carbon_intensity(sec) >= 0

    def test_string_to_sector(self) -> None:
        assert sectoral_carbon_intensity("energy") == sectoral_carbon_intensity(Sector.ENERGY)


class TestCarbonPriceCurve:
    def test_basic_linear(self) -> None:
        path = carbon_price_curve([(0.0, 50.0), (10.0, 300.0)])
        assert path(0.0) == 50.0
        assert path(10.0) == 300.0
        assert path(5.0) == pytest.approx(175.0)

    def test_flat_extrapolation_left(self) -> None:
        path = carbon_price_curve([(5.0, 100.0), (10.0, 200.0)])
        assert path(0.0) == 100.0

    def test_flat_extrapolation_right(self) -> None:
        path = carbon_price_curve([(0.0, 50.0), (10.0, 300.0)])
        assert path(50.0) == 300.0

    def test_unsorted_knots_sorted_internally(self) -> None:
        path = carbon_price_curve([(10.0, 300.0), (0.0, 50.0)])
        assert path(5.0) == pytest.approx(175.0)

    def test_negative_prices_rejected(self) -> None:
        with pytest.raises(MertonInputError):
            carbon_price_curve([(0.0, -10.0), (10.0, 100.0)])

    def test_empty_knots_rejected(self) -> None:
        with pytest.raises(MertonInputError):
            carbon_price_curve([])

    def test_single_knot(self) -> None:
        path = carbon_price_curve([(5.0, 100.0)])
        assert path(0.0) == 100.0
        assert path(100.0) == 100.0


class TestCarbonPriceToWritedown:
    def test_zero_price_zero_writedown(self) -> None:
        assert carbon_price_to_writedown(0.0, Sector.ENERGY) == 0.0

    def test_full_pass_through_no_writedown(self) -> None:
        w = carbon_price_to_writedown(500.0, Sector.ENERGY, pass_through=1.0)
        assert w == 0.0

    def test_zero_pass_through_full_hit(self) -> None:
        w = carbon_price_to_writedown(500.0, Sector.ENERGY, pass_through=0.0)
        # 500 * 900 / 1e6 = 0.45
        assert w == pytest.approx(0.45)

    def test_partial_pass_through(self) -> None:
        w = carbon_price_to_writedown(500.0, Sector.ENERGY, pass_through=0.5)
        # 0.5 * 500 * 900 / 1e6 = 0.225
        assert w == pytest.approx(0.225)

    def test_clamped_to_one(self) -> None:
        # Outlandish carbon price → writedown caps at 1.0.
        w = carbon_price_to_writedown(1e6, Sector.ENERGY, pass_through=0.0)
        assert w == 1.0

    def test_override_intensity(self) -> None:
        w = carbon_price_to_writedown(100.0, Sector.ENERGY, pass_through=0.0, intensity=100.0)
        # 100 * 100 / 1e6 = 0.01
        assert w == pytest.approx(0.01)

    def test_invalid_pass_through(self) -> None:
        with pytest.raises(MertonInputError):
            carbon_price_to_writedown(50.0, Sector.ENERGY, pass_through=-0.1)
        with pytest.raises(MertonInputError):
            carbon_price_to_writedown(50.0, Sector.ENERGY, pass_through=1.1)

    def test_invalid_carbon_price(self) -> None:
        with pytest.raises(MertonInputError):
            carbon_price_to_writedown(-50.0, Sector.ENERGY)

    def test_invalid_intensity(self) -> None:
        with pytest.raises(MertonInputError):
            carbon_price_to_writedown(50.0, Sector.ENERGY, intensity=-1.0)

    def test_sector_ranking(self) -> None:
        # Same price, lower-intensity sector → smaller writedown.
        w_energy = carbon_price_to_writedown(100.0, Sector.ENERGY, pass_through=0.0)
        w_tech = carbon_price_to_writedown(100.0, Sector.TECH, pass_through=0.0)
        assert w_tech < w_energy


@pytest.fixture
def base_scenario() -> ClimateScenario:
    return ClimateScenario(
        name="Test",
        carbon_price_path=carbon_price_curve([(0.0, 50.0), (10.0, 300.0)]),
        pd_multipliers={Sector.ENERGY: 2.0, Sector.TECH: 0.95},
        pass_through=0.5,
        physical_writedown=0.001,
    )


@pytest.fixture
def firm() -> Firm:
    return Firm(
        equity=100.0,
        debt_short=10.0,
        debt_long=30.0,
        equity_vol=0.30,
        horizon=5.0,
        ticker="ACME",
    )


class TestClimateScenario:
    def test_carbon_price_at_horizon(self, base_scenario: ClimateScenario) -> None:
        assert base_scenario.carbon_price(5.0) == pytest.approx(175.0)

    def test_negative_horizon_invalid(self, base_scenario: ClimateScenario) -> None:
        with pytest.raises(MertonInputError):
            base_scenario.carbon_price(-1.0)

    def test_writedown_combines_transition_and_physical(
        self, base_scenario: ClimateScenario
    ) -> None:
        w = base_scenario.asset_writedown(10.0, Sector.ENERGY)
        # transition = 0.5 * 300 * 900 / 1e6 = 0.135; physical = 0.001*10 = 0.01;
        # total = 0.135 + (1-0.135) * 0.01 = 0.14365
        assert w == pytest.approx(0.14365, rel=1e-6)

    def test_pd_multiplier_lookup(self, base_scenario: ClimateScenario) -> None:
        assert base_scenario.pd_multiplier(Sector.ENERGY) == 2.0
        assert base_scenario.pd_multiplier(Sector.TECH) == 0.95
        # Unspecified sector defaults to 1.0.
        assert base_scenario.pd_multiplier(Sector.HEALTHCARE) == 1.0

    def test_pd_multiplier_accepts_string(self, base_scenario: ClimateScenario) -> None:
        assert base_scenario.pd_multiplier("energy") == 2.0

    def test_apply_returns_writedown_equity(
        self, base_scenario: ClimateScenario, firm: Firm
    ) -> None:
        res = base_scenario.apply(firm, sector=Sector.ENERGY)
        assert res.firm.equity < firm.equity
        assert res.parameters["writedown"] > 0
        assert res.parameters["sector"] == "Sector.ENERGY"
        assert res.parameters["pd_multiplier"] == 2.0

    def test_apply_requires_sector(self, base_scenario: ClimateScenario, firm: Firm) -> None:
        with pytest.raises(MertonInputError, match="sector"):
            base_scenario.apply(firm)

    def test_string_keys_in_pd_multipliers(self) -> None:
        # Constructor must normalise string keys to the Sector enum.
        s = ClimateScenario(
            name="x",
            carbon_price_path=carbon_price_curve([(0.0, 100.0)]),
            pd_multipliers={"energy": 1.5},
        )
        assert s.pd_multipliers[Sector.ENERGY] == 1.5

    def test_invalid_pass_through_rejected(self) -> None:
        with pytest.raises(MertonInputError):
            ClimateScenario(
                name="x",
                carbon_price_path=carbon_price_curve([(0.0, 100.0)]),
                pass_through=1.5,
            )

    def test_negative_physical_writedown_rejected(self) -> None:
        with pytest.raises(MertonInputError):
            ClimateScenario(
                name="x",
                carbon_price_path=carbon_price_curve([(0.0, 100.0)]),
                physical_writedown=-0.01,
            )

    def test_negative_pd_multiplier_rejected(self) -> None:
        with pytest.raises(MertonInputError):
            ClimateScenario(
                name="x",
                carbon_price_path=carbon_price_curve([(0.0, 100.0)]),
                pd_multipliers={Sector.ENERGY: -0.1},
            )

    def test_apply_with_intensity_override(
        self, base_scenario: ClimateScenario, firm: Firm
    ) -> None:
        res_default = base_scenario.apply(firm, sector=Sector.ENERGY)
        res_override = base_scenario.apply(firm, sector=Sector.ENERGY, intensity=1.0)
        # Override to a tiny intensity makes the writedown smaller.
        assert res_override.parameters["writedown"] < res_default.parameters["writedown"]

    def test_extreme_writedown_keeps_positive_equity(self, firm: Firm) -> None:
        wipeout = ClimateScenario(
            name="wipeout",
            carbon_price_path=carbon_price_curve([(0.0, 1.0e9)]),
            pass_through=0.0,
            physical_writedown=0.0,
        )
        res = wipeout.apply(firm, sector=Sector.ENERGY)
        assert res.firm.equity > 0  # never zero — clamped to 1e-6 minimum
