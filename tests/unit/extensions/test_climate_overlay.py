"""Tests for the ClimateOverlay structural-model wrapper."""

from __future__ import annotations

import pytest

from merton import Firm, MertonModel
from merton.exceptions import MertonInputError
from merton.extensions import (
    BlackCoxModel,
    ClimateOverlay,
    CreditGradesModel,
    LelandToftModel,
)
from merton.scenarios.climate import ClimateScenario, Sector, carbon_price_curve
from merton.scenarios.predefined.ngfs import (
    current_policies,
    delayed_transition,
    fragmented_world,
    net_zero_2050,
)


@pytest.fixture
def firm() -> Firm:
    return Firm(
        equity=100.0,
        debt_short=10.0,
        debt_long=30.0,
        equity_vol=0.30,
        rf=0.04,
        horizon=5.0,
        ticker="ACME",
    )


@pytest.fixture
def mild_scenario() -> ClimateScenario:
    return ClimateScenario(
        name="Mild",
        carbon_price_path=carbon_price_curve([(0.0, 30.0), (5.0, 50.0)]),
        pd_multipliers={Sector.ENERGY: 1.2},
        pass_through=0.6,
        physical_writedown=0.0005,
    )


class TestClimateOverlayBasic:
    def test_fit_returns_structural_result(
        self, firm: Firm, mild_scenario: ClimateScenario
    ) -> None:
        overlay = ClimateOverlay(MertonModel(), scenario=mild_scenario, sector=Sector.ENERGY)
        r = overlay.fit(firm)
        assert r.method == "climate_overlay"
        assert 0 <= r.pd <= 1
        assert r.diagnostics["scenario"] == "Mild"
        assert r.diagnostics["sector"] == "energy"
        assert r.diagnostics["pd_multiplier"] == 1.2

    def test_overlay_increases_pd_relative_to_baseline(
        self, firm: Firm, mild_scenario: ClimateScenario
    ) -> None:
        base = MertonModel()
        baseline = base.fit(firm)
        overlay = ClimateOverlay(base, scenario=mild_scenario, sector=Sector.ENERGY)
        stressed = overlay.fit(firm)
        assert stressed.pd >= baseline.pd

    def test_overlay_dd_less_than_baseline(
        self, firm: Firm, mild_scenario: ClimateScenario
    ) -> None:
        base = MertonModel()
        baseline = base.fit(firm)
        overlay = ClimateOverlay(base, scenario=mild_scenario, sector=Sector.ENERGY)
        stressed = overlay.fit(firm)
        assert stressed.dd <= baseline.dd

    def test_string_sector_normalised(self, firm: Firm, mild_scenario: ClimateScenario) -> None:
        overlay = ClimateOverlay(MertonModel(), scenario=mild_scenario, sector="energy")
        assert overlay.sector == Sector.ENERGY

    def test_base_without_fit_rejected(self, mild_scenario: ClimateScenario) -> None:
        class NotAModel:
            pass

        with pytest.raises(MertonInputError):
            ClimateOverlay(NotAModel(), scenario=mild_scenario, sector=Sector.ENERGY)


class TestClimateOverlayWithExtensions:
    @pytest.mark.parametrize(
        "base",
        [BlackCoxModel(), LelandToftModel(coupon=2.5)],
        ids=["black_cox", "leland_toft"],
    )
    def test_works_with_structural_extension(
        self, firm: Firm, mild_scenario: ClimateScenario, base
    ) -> None:
        overlay = ClimateOverlay(base, scenario=mild_scenario, sector=Sector.ENERGY)
        r = overlay.fit(firm)
        assert 0 <= r.pd <= 1
        assert r.diagnostics["base_method"] in {"black_cox", "leland_toft"}

    def test_with_creditgrades(self, firm: Firm, mild_scenario: ClimateScenario) -> None:
        base = CreditGradesModel(debt_per_share=40.0)
        overlay = ClimateOverlay(base, scenario=mild_scenario, sector=Sector.ENERGY)
        r = overlay.fit(firm)
        assert 0 <= r.pd <= 1


class TestNGFSEndToEnd:
    @pytest.mark.parametrize(
        "scenario_fn",
        [net_zero_2050, delayed_transition, current_policies, fragmented_world],
    )
    def test_runs_to_completion(self, firm: Firm, scenario_fn) -> None:
        sc = scenario_fn()
        overlay = ClimateOverlay(MertonModel(), scenario=sc, sector=Sector.ENERGY)
        r = overlay.fit(firm)
        assert 0 <= r.pd <= 1
        # All four NGFS scenarios penalise energy.
        baseline = MertonModel().fit(firm)
        assert r.pd >= baseline.pd

    def test_high_carbon_path_more_punishing(self, firm: Firm) -> None:
        """Net Zero 2050 ramps carbon faster than Current Policies → higher PD."""
        overlay_nz = ClimateOverlay(MertonModel(), scenario=net_zero_2050(), sector=Sector.ENERGY)
        overlay_cp = ClimateOverlay(
            MertonModel(), scenario=current_policies(), sector=Sector.ENERGY
        )
        assert overlay_nz.fit(firm).pd > overlay_cp.fit(firm).pd

    def test_tech_less_stressed_than_energy(self, firm: Firm) -> None:
        sc = net_zero_2050()
        pd_energy = ClimateOverlay(MertonModel(), scenario=sc, sector=Sector.ENERGY).fit(firm).pd
        pd_tech = ClimateOverlay(MertonModel(), scenario=sc, sector=Sector.TECH).fit(firm).pd
        assert pd_energy > pd_tech


class TestExtremes:
    def test_pd_clamped_to_unit_interval(self, firm: Firm) -> None:
        # Outlandish multiplier: PD must stay ≤ 1.
        s = ClimateScenario(
            name="extreme",
            carbon_price_path=carbon_price_curve([(0.0, 1.0e6)]),
            pd_multipliers={Sector.ENERGY: 1.0e6},
            pass_through=0.0,
        )
        overlay = ClimateOverlay(MertonModel(), scenario=s, sector=Sector.ENERGY)
        r = overlay.fit(firm)
        assert r.pd == 1.0
        assert r.dd == float("-inf")

    def test_zero_carbon_minimal_writedown(self, firm: Firm) -> None:
        s = ClimateScenario(
            name="zero",
            carbon_price_path=carbon_price_curve([(0.0, 0.0)]),
            pd_multipliers={},
            pass_through=0.5,
            physical_writedown=0.0,
        )
        overlay = ClimateOverlay(MertonModel(), scenario=s, sector=Sector.ENERGY)
        baseline = MertonModel().fit(firm)
        r = overlay.fit(firm)
        # No carbon, no physical risk, no multiplier override → ~baseline PD.
        assert r.pd == pytest.approx(baseline.pd, rel=1e-6)
