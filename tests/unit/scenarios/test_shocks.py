"""Tests for atomic firm-level shocks."""

from __future__ import annotations

import pytest

from merton import Firm
from merton.exceptions import MertonInputError
from merton.scenarios import (
    CompositeScenario,
    Scenario,
    ScenarioResult,
    debt_shock,
    equity_shock,
    rate_shock,
    vol_shock,
)


@pytest.fixture
def firm() -> Firm:
    return Firm(
        equity=100.0,
        debt_short=10.0,
        debt_long=30.0,
        equity_vol=0.25,
        rf=0.04,
        horizon=1.0,
        ticker="ACME",
    )


class TestEquityShock:
    def test_basic_drawdown(self, firm: Firm) -> None:
        res = equity_shock(factor=0.7).apply(firm)
        assert res.firm.equity == pytest.approx(70.0)
        assert res.scenario == "equity_shock"
        assert res.parameters == {"factor": 0.7}

    def test_rally(self, firm: Firm) -> None:
        res = equity_shock(factor=1.5).apply(firm)
        assert res.firm.equity == pytest.approx(150.0)

    def test_invalid_factor(self) -> None:
        with pytest.raises(MertonInputError):
            equity_shock(factor=0.0)
        with pytest.raises(MertonInputError):
            equity_shock(factor=-0.1)

    def test_returns_scenario_result(self, firm: Firm) -> None:
        res = equity_shock(factor=0.5).apply(firm)
        assert isinstance(res, ScenarioResult)
        assert res.description  # non-empty

    def test_firm_immutable(self, firm: Firm) -> None:
        original = firm.equity
        equity_shock(factor=0.5).apply(firm)
        assert firm.equity == original  # apply returns new firm


class TestVolShock:
    def test_basic(self, firm: Firm) -> None:
        res = vol_shock(factor=2.0).apply(firm)
        assert res.firm.equity_vol == pytest.approx(0.50)

    def test_invalid_factor(self) -> None:
        with pytest.raises(MertonInputError):
            vol_shock(factor=-1.0)

    def test_requires_equity_vol(self) -> None:
        firm = Firm(equity=100.0, debt_short=10.0, debt_long=20.0)  # equity_vol=None
        with pytest.raises(MertonInputError, match="equity_vol"):
            vol_shock(factor=1.5).apply(firm)


class TestRateShock:
    def test_tightening(self, firm: Firm) -> None:
        res = rate_shock(delta=0.02).apply(firm)
        assert res.firm.rf == pytest.approx(0.06)
        assert "+200" in res.description or "+200 bps" in res.description

    def test_easing(self, firm: Firm) -> None:
        res = rate_shock(delta=-0.01).apply(firm)
        assert res.firm.rf == pytest.approx(0.03)


class TestDebtShock:
    def test_uniform_factor(self, firm: Firm) -> None:
        res = debt_shock(factor=1.5).apply(firm)
        assert res.firm.debt_short == pytest.approx(15.0)
        assert res.firm.debt_long == pytest.approx(45.0)

    def test_split_factors(self, firm: Firm) -> None:
        res = debt_shock(short_factor=2.0, long_factor=0.8).apply(firm)
        assert res.firm.debt_short == pytest.approx(20.0)
        assert res.firm.debt_long == pytest.approx(24.0)

    def test_negative_factor_invalid(self) -> None:
        with pytest.raises(MertonInputError):
            debt_shock(factor=-0.1)
        with pytest.raises(MertonInputError):
            debt_shock(short_factor=-0.1, long_factor=1.0)


class TestComposite:
    def test_pipe_operator(self, firm: Firm) -> None:
        composite = equity_shock(factor=0.5) | vol_shock(factor=2.0)
        assert isinstance(composite, CompositeScenario)
        res = composite.apply(firm)
        assert res.firm.equity == pytest.approx(50.0)
        assert res.firm.equity_vol == pytest.approx(0.50)
        # Composite scenario name reflects the stack.
        assert "equity_shock" in res.scenario
        assert "vol_shock" in res.scenario

    def test_three_way_pipeline(self, firm: Firm) -> None:
        composite = equity_shock(factor=0.6) | vol_shock(factor=1.5) | rate_shock(delta=0.01)
        res = composite.apply(firm)
        assert res.firm.equity == pytest.approx(60.0)
        assert res.firm.equity_vol == pytest.approx(0.375)
        assert res.firm.rf == pytest.approx(0.05)

    def test_parameters_accumulate(self, firm: Firm) -> None:
        composite = equity_shock(factor=0.5) | vol_shock(factor=2.0)
        res = composite.apply(firm)
        assert "equity_shock" in res.parameters
        assert "vol_shock" in res.parameters


class TestScenarioABC:
    def test_subclass_must_implement_apply(self) -> None:
        with pytest.raises(TypeError):
            Scenario()  # type: ignore[abstract]

    def test_repr_includes_name(self, firm: Firm) -> None:
        rep = repr(equity_shock(factor=0.5))
        assert "equity_shock" in rep
