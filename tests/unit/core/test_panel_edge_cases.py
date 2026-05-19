"""Edge cases for FirmPanel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from merton import Firm, FirmPanel
from merton.exceptions import MertonInputError


@pytest.fixture
def panel() -> FirmPanel:
    return FirmPanel.from_dict(
        {
            "ticker": ["A", "B", "C"],
            "equity": [100.0, 200.0, 150.0],
            "debt_short": [20.0, 40.0, 30.0],
            "debt_long": [30.0, 60.0, 45.0],
            "equity_vol": [0.30, 0.25, 0.35],
            "dividend_yield": [0.0, 0.01, 0.02],
            "horizon": [1.0, 2.0, 1.0],
        }
    )


class TestSlicingErrors:
    def test_slice_with_step_raises(self, panel: FirmPanel) -> None:
        with pytest.raises(MertonInputError):
            _ = panel[::2]

    def test_wrong_mask_shape_raises(self, panel: FirmPanel) -> None:
        with pytest.raises(MertonInputError):
            _ = panel[np.array([True, False])]  # too short

    def test_non_bool_mask_raises(self, panel: FirmPanel) -> None:
        with pytest.raises(MertonInputError):
            _ = panel[np.array([1, 0, 1])]

    def test_out_of_range_index_raises(self, panel: FirmPanel) -> None:
        with pytest.raises(IndexError):
            _ = panel[99]


class TestFirmIterationFields:
    def test_full_field_propagation(self, panel: FirmPanel) -> None:
        firm = panel[1]
        assert isinstance(firm, Firm)
        assert firm.ticker == "B"
        assert firm.equity == 200.0
        assert firm.equity_vol == 0.25
        assert firm.dividend_yield == 0.01
        assert firm.horizon == 2.0


class TestRoundTrips:
    def test_arrow_round_trip(self, panel: FirmPanel) -> None:
        table = panel.to_arrow()
        assert isinstance(table, pa.Table)
        re_panel = FirmPanel.from_arrow(table)
        assert re_panel.columns == panel.columns

    def test_pandas_round_trip(self, panel: FirmPanel) -> None:
        df = panel.to_pandas()
        re_panel = FirmPanel.from_pandas(df)
        assert len(re_panel) == len(panel)

    def test_repr_and_html(self, panel: FirmPanel) -> None:
        text = repr(panel)
        assert "FirmPanel" in text
        html = panel._repr_html_()
        assert "<table" in html


class TestRenamingMapping:
    def test_from_pandas_with_mapping(self) -> None:
        raw = pd.DataFrame(
            {
                "mkt_cap": [100.0, 200.0],
                "st_debt": [20.0, 40.0],
                "lt_debt": [30.0, 60.0],
            }
        )
        panel = FirmPanel.from_pandas(
            raw, mapping={"mkt_cap": "equity", "st_debt": "debt_short", "lt_debt": "debt_long"}
        )
        assert "equity" in panel.columns
        assert "debt_short" in panel.columns
