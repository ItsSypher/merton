"""Tests for the packaged NGFS Phase V scenarios."""

from __future__ import annotations

import pytest

from merton.scenarios.climate import ClimateScenario, Sector
from merton.scenarios.predefined.ngfs import (
    HEADLINE_SCENARIOS,
    current_policies,
    delayed_transition,
    fragmented_world,
    net_zero_2050,
)

SCENARIOS = [net_zero_2050, delayed_transition, current_policies, fragmented_world]


class TestHeadlineRegistry:
    def test_all_four_scenarios_registered(self) -> None:
        assert set(HEADLINE_SCENARIOS) == {
            "net_zero_2050",
            "delayed_transition",
            "current_policies",
            "fragmented_world",
        }

    def test_registry_factory_returns_scenario(self) -> None:
        for fn in HEADLINE_SCENARIOS.values():
            sc = fn()
            assert isinstance(sc, ClimateScenario)
            assert sc.name


class TestCarbonPricePaths:
    @pytest.mark.parametrize("scenario_fn", SCENARIOS)
    def test_carbon_price_non_decreasing_in_year(self, scenario_fn) -> None:
        sc = scenario_fn()
        years = [0.0, 5.0, 10.0, 20.0, 30.0]
        prices = [sc.carbon_price(t) for t in years]
        # Monotone non-decreasing — all four published scenarios have rising
        # carbon-price paths over the standard 30-year horizon.
        assert all(prices[i] <= prices[i + 1] for i in range(len(prices) - 1))

    def test_net_zero_dominates_current_policies_at_horizon(self) -> None:
        assert net_zero_2050().carbon_price(20.0) > current_policies().carbon_price(20.0)

    def test_delayed_transition_overtakes_current_policies(self) -> None:
        # Delayed transition starts low, then spikes — at 2050 it should
        # exceed Current Policies.
        assert delayed_transition().carbon_price(30.0) > current_policies().carbon_price(30.0)

    def test_fragmented_world_between_net_zero_and_current(self) -> None:
        nz = net_zero_2050().carbon_price(30.0)
        cp = current_policies().carbon_price(30.0)
        fw = fragmented_world().carbon_price(30.0)
        assert cp < fw < nz


class TestPDMultipliers:
    @pytest.mark.parametrize("scenario_fn", SCENARIOS)
    def test_high_intensity_sectors_have_high_multiplier(self, scenario_fn) -> None:
        sc = scenario_fn()
        assert sc.pd_multiplier(Sector.ENERGY) >= sc.pd_multiplier(Sector.TECH)

    def test_delayed_transition_more_punishing_than_net_zero(self) -> None:
        # The whole point of the delayed scenario: abrupt repricing → higher
        # PD multipliers on transition-exposed sectors than the orderly path.
        assert delayed_transition().pd_multiplier(Sector.ENERGY) > net_zero_2050().pd_multiplier(
            Sector.ENERGY
        )

    def test_current_policies_emphasises_real_estate(self) -> None:
        # Physical-risk dominated scenario should hit real-estate harder
        # than the transition-driven ones.
        assert current_policies().pd_multiplier(Sector.REAL_ESTATE) > net_zero_2050().pd_multiplier(
            Sector.REAL_ESTATE
        )


class TestPhysicalRisk:
    def test_current_policies_higher_physical_than_net_zero(self) -> None:
        assert current_policies().physical_writedown > net_zero_2050().physical_writedown

    def test_all_physical_writedowns_non_negative(self) -> None:
        for fn in SCENARIOS:
            assert fn().physical_writedown >= 0.0
