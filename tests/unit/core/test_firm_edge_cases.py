"""Firm-specific edge cases (from_dict, from_yfinance, immutable replace)."""

from __future__ import annotations

import numpy as np
import pytest

from merton import Firm
from merton.core.default_point import DefaultPoint
from merton.exceptions import MertonInputError


class TestFirmConstruction:
    def test_from_dict(self) -> None:
        firm = Firm.from_dict(
            {"equity": 100.0, "debt_short": 20.0, "debt_long": 30.0, "equity_vol": 0.30}
        )
        assert firm.equity == 100.0
        assert firm.equity_vol == 0.30

    def test_rejects_negative_horizon(self) -> None:
        with pytest.raises(MertonInputError):
            Firm(equity=100.0, debt_short=20.0, debt_long=30.0, horizon=-1.0)

    def test_total_debt_property(self) -> None:
        firm = Firm(equity=100.0, debt_short=20.0, debt_long=30.0)
        assert firm.total_debt == 50.0

    def test_replace_preserves_other_fields(self) -> None:
        firm = Firm(
            equity=100.0,
            debt_short=20.0,
            debt_long=30.0,
            equity_vol=0.30,
            ticker="X",
        )
        new = firm.replace(horizon=5.0)
        assert new.horizon == 5.0
        assert new.equity_vol == 0.30
        assert new.ticker == "X"


class TestFromYFinance:
    def test_missing_yfinance_falls_back_to_import_error(self, monkeypatch) -> None:
        # Inject a fake ImportError to simulate the missing extra.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "yfinance":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError):
            Firm.from_yfinance("AAPL")

    def test_uses_yfinance_mock(self, monkeypatch) -> None:
        # Build a fake yfinance module with the minimal surface that Firm needs.
        class FakeBalanceSheet:
            def __init__(self):
                self._data = {
                    "Current Debt": [20e9],
                    "Long Term Debt": [90e9],
                }

            def loc(self):  # not used; we patch __getitem__-style access
                return self

        class FakeFastInfo(dict):
            def get(self, k, default=None):  # type: ignore[override]
                return super().get(k, default)

        class FakeTicker:
            def __init__(self, _ticker):
                self.fast_info = FakeFastInfo(market_cap=3.0e12)
                import pandas as pd

                idx = pd.date_range("2024-01-01", periods=30, freq="B")
                close = np.linspace(100.0, 105.0, 30)
                self._df = pd.DataFrame({"Close": close}, index=idx)

                class _BS:
                    def __init__(self):
                        self._frame = pd.DataFrame({0: [20e9]}, index=["Current Debt"])

                    @property
                    def loc(self):
                        return self._frame.loc

                self.balance_sheet = _BS()

            def history(self, period="2y", auto_adjust=False):
                return self._df

        import sys
        import types

        fake_yf = types.ModuleType("yfinance")
        fake_yf.Ticker = FakeTicker
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)

        firm = Firm.from_yfinance("AAPL")
        assert firm.ticker == "AAPL"
        assert firm.equity == 3.0e12
        assert firm.equity_vol > 0
        assert firm.debt_short == 20e9


class TestDefaultPointCustom:
    def test_custom_callable_used(self) -> None:
        firm = Firm(
            equity=100,
            debt_short=20,
            debt_long=30,
            equity_vol=0.30,
            default_point=DefaultPoint.CUSTOM,
            custom_default_point=lambda st, lt: st + 0.75 * lt,
        )
        assert float(firm.default_point_value()) == 42.5
