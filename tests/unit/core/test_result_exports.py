"""Coverage for MertonResult export helpers (.to_polars, .to_excel)."""

from __future__ import annotations

import pandas as pd
import pytest

from merton import Firm, fit


@pytest.fixture
def fitted():
    firm = Firm(
        equity=100.0,
        debt_short=20.0,
        debt_long=30.0,
        equity_vol=0.30,
        rf=0.04,
        horizon=1.0,
        ticker="TEST",
    )
    return fit(firm, method="jmr_iterative")


class TestResultExports:
    def test_to_excel_writes_workbook(self, fitted, tmp_path) -> None:
        path = tmp_path / "result.xlsx"
        try:
            fitted.to_excel(str(path))
        except ImportError:
            pytest.skip("openpyxl not installed in this env")
        assert path.exists()
        # Read back and confirm headers.
        sheets = pd.read_excel(path, sheet_name=None)
        assert "Merton" in sheets
        assert "TermStructure" in sheets

    def test_to_polars_or_import_skip(self, fitted) -> None:
        try:
            df = fitted.to_polars()
        except ImportError:
            pytest.skip("polars not installed")
        assert len(df) == 1
