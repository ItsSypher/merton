"""Tests for batch_fit panel calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from merton import FirmPanel, batch_fit


@pytest.fixture
def panel_df() -> pd.DataFrame:
    rng = np.random.default_rng(101)
    n = 20
    return pd.DataFrame(
        {
            "ticker": [f"F{i:02d}" for i in range(n)],
            "equity": rng.uniform(80, 250, n),
            "debt_short": rng.uniform(10, 40, n),
            "debt_long": rng.uniform(20, 60, n),
            "equity_vol": rng.uniform(0.20, 0.45, n),
            "rf": np.full(n, 0.04),
        }
    )


class TestBatchFit:
    def test_pandas_in_pandas_out(self, panel_df: pd.DataFrame) -> None:
        out = batch_fit(panel_df, method="jmr_iterative", n_jobs=2)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == len(panel_df)
        for col in ("dd", "pd", "asset_value", "asset_vol", "method", "ticker"):
            assert col in out.columns
        assert out["converged"].all()

    def test_panel_in(self, panel_df: pd.DataFrame) -> None:
        panel = FirmPanel.from_pandas(panel_df)
        out = batch_fit(panel, method="jmr_iterative", n_jobs=2)
        assert len(out) == len(panel)

    def test_arrow_in_arrow_out(self, panel_df: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(panel_df, preserve_index=False)
        out = batch_fit(table, method="jmr_iterative", n_jobs=2)
        assert isinstance(out, pa.Table)
        assert out.num_rows == len(panel_df)

    def test_sequential_dispatch(self, panel_df: pd.DataFrame) -> None:
        out = batch_fit(panel_df, method="naive", dispatch="sequential", n_jobs=1)
        assert len(out) == len(panel_df)

    def test_unknown_method_warns(self, panel_df: pd.DataFrame) -> None:
        with pytest.warns(UserWarning):
            out = batch_fit(
                panel_df, method="jmr_iterative", n_jobs=1, on_error="warn", tol=1e-50, max_iter=1
            )
        # Either converged=False rows or NaNs.
        assert len(out) == len(panel_df)

    def test_horizon_override(self, panel_df: pd.DataFrame) -> None:
        out = batch_fit(panel_df, method="jmr_iterative", n_jobs=2, horizon=5.0)
        assert all(out["horizon"] == 5.0)

    def test_empty_input_returns_empty(self) -> None:
        empty = pd.DataFrame(
            {
                "equity": pd.Series([], dtype=float),
                "debt_short": pd.Series([], dtype=float),
                "debt_long": pd.Series([], dtype=float),
            }
        )
        out = batch_fit(empty, method="naive")
        assert len(out) == 0
