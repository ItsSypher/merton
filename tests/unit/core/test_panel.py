"""Tests for the FirmPanel Arrow-backed container."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from merton import Firm, FirmPanel
from merton.exceptions import MertonInputError


@pytest.fixture
def basic_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "equity": [100.0, 200.0, 150.0],
            "debt_short": [20.0, 40.0, 30.0],
            "debt_long": [30.0, 60.0, 45.0],
            "equity_vol": [0.30, 0.25, 0.35],
            "rf": [0.04, 0.04, 0.04],
        }
    )


class TestFirmPanel:
    def test_from_pandas(self, basic_df: pd.DataFrame) -> None:
        panel = FirmPanel.from_pandas(basic_df)
        assert len(panel) == 3
        assert set(panel.columns) >= {"equity", "debt_short", "debt_long"}

    def test_from_dict(self) -> None:
        panel = FirmPanel.from_dict(
            {"equity": [100.0, 200.0], "debt_short": [20.0, 40.0], "debt_long": [30.0, 60.0]}
        )
        assert len(panel) == 2

    def test_missing_required_column_raises(self) -> None:
        with pytest.raises(MertonInputError):
            FirmPanel.from_dict({"equity": [100.0], "debt_short": [20.0]})

    def test_iter_yields_firms(self, basic_df: pd.DataFrame) -> None:
        panel = FirmPanel.from_pandas(basic_df)
        firms = list(panel.firms())
        assert len(firms) == 3
        assert isinstance(firms[0], Firm)
        assert firms[0].ticker == "A"
        assert firms[0].equity == 100.0

    def test_indexing_int(self, basic_df: pd.DataFrame) -> None:
        panel = FirmPanel.from_pandas(basic_df)
        firm = panel[1]
        assert isinstance(firm, Firm)
        assert firm.ticker == "B"
        # Negative index works too.
        assert panel[-1].ticker == "C"

    def test_slicing(self, basic_df: pd.DataFrame) -> None:
        panel = FirmPanel.from_pandas(basic_df)
        sub = panel[1:3]
        assert isinstance(sub, FirmPanel)
        assert len(sub) == 2
        assert sub[0].ticker == "B"

    def test_boolean_mask(self, basic_df: pd.DataFrame) -> None:
        panel = FirmPanel.from_pandas(basic_df)
        mask = panel.equity > 120
        sub = panel[mask]
        assert len(sub) == 2  # firms B and C

    def test_head(self, basic_df: pd.DataFrame) -> None:
        panel = FirmPanel.from_pandas(basic_df)
        head = panel.head(2)
        assert len(head) == 2

    def test_round_trip_pandas(self, basic_df: pd.DataFrame) -> None:
        panel = FirmPanel.from_pandas(basic_df)
        df = panel.to_pandas()
        assert set(df.columns) >= set(basic_df.columns)
        assert len(df) == len(basic_df)

    def test_arrow_round_trip(self, basic_df: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(basic_df, preserve_index=False)
        panel = FirmPanel.from_arrow(table)
        round_trip = panel.to_arrow()
        assert isinstance(round_trip, pa.Table)
        assert round_trip.num_rows == 3

    def test_columnar_accessors(self, basic_df: pd.DataFrame) -> None:
        panel = FirmPanel.from_pandas(basic_df)
        assert isinstance(panel.equity, np.ndarray)
        assert panel.equity[0] == 100.0
        assert panel.equity_vol is not None and panel.equity_vol[1] == 0.25

    def test_csv_and_parquet_round_trip(self, basic_df: pd.DataFrame, tmp_path) -> None:
        csv_path = tmp_path / "panel.csv"
        parquet_path = tmp_path / "panel.parquet"
        panel = FirmPanel.from_pandas(basic_df)
        panel.to_csv(csv_path)
        panel.to_parquet(parquet_path)
        assert FirmPanel.from_csv(csv_path).columns == panel.columns
        assert FirmPanel.from_parquet(parquet_path).columns == panel.columns
