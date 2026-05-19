"""Tests for the rolling-window backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from merton.backtest import rolling_window
from merton.exceptions import MertonInputError


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    n = 400
    pd_scores = np.clip(rng.uniform(0, 0.4, n), 0, 1)
    defaults = (rng.uniform(0, 1, n) < pd_scores).astype(int)
    return pd.DataFrame({"date": dates, "pd": pd_scores, "default": defaults})


class TestRollingWindow:
    def test_basic_run(self, panel: pd.DataFrame) -> None:
        result = rolling_window(panel, window="60D", step="30D")
        df = result.to_pandas()
        assert "auc" in df.columns
        assert len(df) > 0

    def test_missing_column_raises(self, panel: pd.DataFrame) -> None:
        with pytest.raises(MertonInputError):
            rolling_window(panel.drop(columns=["pd"]))
