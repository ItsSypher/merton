"""Tests for the CreditGrades model."""

from __future__ import annotations

import numpy as np
import pytest

from merton import Firm
from merton.exceptions import MertonInputError
from merton.extensions import (
    CreditGradesModel,
    creditgrades_pd,
    creditgrades_spread,
    creditgrades_survival,
)


class TestCreditGradesSurvival:
    def test_well_capitalised_firm_high_survival(self) -> None:
        s = float(creditgrades_survival(equity=100, equity_vol=0.30, debt_per_share=50, T=1.0))
        assert 0.95 < s <= 1.0

    def test_leveraged_firm_lower_survival(self) -> None:
        s_low = float(creditgrades_survival(equity=100, equity_vol=0.30, debt_per_share=50, T=1.0))
        s_hi = float(creditgrades_survival(equity=30, equity_vol=0.50, debt_per_share=70, T=1.0))
        assert s_hi < s_low

    def test_survival_pd_complement(self) -> None:
        s = float(creditgrades_survival(equity=50, equity_vol=0.40, debt_per_share=60, T=2.0))
        p = float(creditgrades_pd(equity=50, equity_vol=0.40, debt_per_share=60, T=2.0))
        assert s + p == pytest.approx(1.0, abs=1e-12)

    def test_longer_horizon_lower_survival(self) -> None:
        s_short = float(creditgrades_survival(50, 0.40, 60, 0.5))
        s_long = float(creditgrades_survival(50, 0.40, 60, 5.0))
        assert s_long < s_short

    def test_higher_vol_lower_survival(self) -> None:
        s_low = float(creditgrades_survival(50, 0.20, 60, 1.0))
        s_hi = float(creditgrades_survival(50, 0.60, 60, 1.0))
        assert s_hi < s_low

    def test_vectorized(self) -> None:
        eqs = np.array([100.0, 50.0, 30.0])
        s = creditgrades_survival(eqs, 0.40, 60.0, 1.0)
        assert s.shape == (3,)
        # Less equity ⇒ less survival.
        assert s[0] > s[1] > s[2]


class TestCreditGradesSpread:
    def test_returns_basis_points(self) -> None:
        bps = float(creditgrades_spread(50, 0.40, 60, 1.0, lgd=0.6))
        bps_dec = float(creditgrades_spread(50, 0.40, 60, 1.0, lgd=0.6, in_bps=False))
        assert bps == pytest.approx(bps_dec * 10_000.0, rel=1e-9)
        assert bps > 0

    def test_higher_lgd_higher_spread(self) -> None:
        a = float(creditgrades_spread(50, 0.40, 60, 1.0, lgd=0.3))
        b = float(creditgrades_spread(50, 0.40, 60, 1.0, lgd=0.9))
        assert b > a


class TestInputValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"equity": 0.0, "equity_vol": 0.3, "debt_per_share": 50.0, "T": 1.0},
            {"equity": 100.0, "equity_vol": 0.0, "debt_per_share": 50.0, "T": 1.0},
            {"equity": 100.0, "equity_vol": 0.3, "debt_per_share": 0.0, "T": 1.0},
            {"equity": 100.0, "equity_vol": 0.3, "debt_per_share": 50.0, "T": 0.0},
        ],
    )
    def test_invalid_inputs_raise(self, kwargs) -> None:
        with pytest.raises(MertonInputError):
            creditgrades_survival(**kwargs)

    def test_invalid_lbar_lam(self) -> None:
        with pytest.raises(MertonInputError):
            creditgrades_survival(100, 0.3, 50, 1.0, lbar=1.5)
        with pytest.raises(MertonInputError):
            creditgrades_survival(100, 0.3, 50, 1.0, lam=-0.1)


class TestCreditGradesModel:
    def test_fit_smoke(self) -> None:
        firm = Firm(
            equity=100.0, debt_short=20, debt_long=30, equity_vol=0.30, rf=0.04, horizon=1.0
        )
        result = CreditGradesModel(debt_per_share=50.0).fit(firm)
        assert result.method == "creditgrades"
        assert 0.0 <= result.pd <= 1.0
        assert result.diagnostics["implied_cds_spread_bps"] >= 0

    def test_requires_equity_vol(self) -> None:
        firm = Firm(equity=100.0, debt_short=20, debt_long=30)
        with pytest.raises(MertonInputError):
            CreditGradesModel().fit(firm)

    def test_default_debt_uses_default_point(self) -> None:
        firm = Firm(equity=100.0, debt_short=20, debt_long=30, equity_vol=0.30)
        result = CreditGradesModel().fit(firm)  # no debt_per_share ⇒ uses default_point
        assert result.diagnostics["debt_per_share"] == pytest.approx(
            float(firm.default_point_value())
        )
