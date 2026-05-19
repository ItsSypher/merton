"""Tests for the Firm dataclass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from merton import Firm
from merton.core.default_point import DefaultPoint
from merton.exceptions import MertonInputError


class TestFirm:
    def test_basic_construction(self, simple_firm: Firm) -> None:
        assert simple_firm.equity == 100.0
        assert simple_firm.total_debt == 50.0

    def test_default_point_kmv(self) -> None:
        firm = Firm(equity=100, debt_short=20, debt_long=30, equity_vol=0.3)
        assert float(firm.default_point_value()) == 35.0  # 20 + 0.5*30

    def test_default_point_total(self) -> None:
        firm = Firm(
            equity=100,
            debt_short=20,
            debt_long=30,
            equity_vol=0.3,
            default_point=DefaultPoint.TOTAL,
        )
        assert float(firm.default_point_value()) == 50.0

    def test_default_point_short_only(self) -> None:
        firm = Firm(
            equity=100,
            debt_short=20,
            debt_long=30,
            equity_vol=0.3,
            default_point=DefaultPoint.SHORT_ONLY,
        )
        assert float(firm.default_point_value()) == 20.0

    def test_default_point_custom(self) -> None:
        firm = Firm(
            equity=100,
            debt_short=20,
            debt_long=30,
            equity_vol=0.3,
            default_point="custom",
            custom_default_point=lambda st, lt: st + 0.75 * lt,
        )
        assert float(firm.default_point_value()) == 42.5

    def test_rejects_zero_equity(self) -> None:
        with pytest.raises(MertonInputError):
            Firm(equity=0.0, debt_short=20, debt_long=30, equity_vol=0.3)

    def test_rejects_negative_debt(self) -> None:
        with pytest.raises(MertonInputError):
            Firm(equity=100, debt_short=-1, debt_long=30, equity_vol=0.3)

    def test_replace_is_immutable(self, simple_firm: Firm) -> None:
        new = simple_firm.replace(horizon=5.0)
        assert simple_firm.horizon == 1.0
        assert new.horizon == 5.0

    def test_from_panel(self) -> None:
        df = pd.DataFrame(
            {
                "equity": [100.0, 110.0],
                "debt_short": [20.0, 22.0],
                "debt_long": [30.0, 33.0],
                "equity_vol": [0.30, 0.28],
            }
        )
        firm = Firm.from_panel(df)
        assert np.array_equal(firm.equity, df["equity"].to_numpy())

    def test_invalid_default_point_kind(self) -> None:
        with pytest.raises(MertonInputError):
            Firm(
                equity=100,
                debt_short=20,
                debt_long=30,
                equity_vol=0.3,
                default_point="bogus",
            ).default_point_value()
