"""Tests for the HTML backtest-report renderer."""

from __future__ import annotations

import numpy as np

from merton.backtest import Backtest
from merton.reports import render_backtest_report


def _make_result():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, 500).astype(float)
    s = np.clip(y * 0.4 + rng.normal(0, 0.15, 500), 0, 1)
    return Backtest().run(s, y, n_bins=10)


class TestHTMLReport:
    def test_returns_html_string(self) -> None:
        html = render_backtest_report(_make_result())
        assert "<html" in html
        assert "AUC" in html or "auc" in html
        assert "<svg" in html  # ROC + calibration plotted

    def test_writes_file(self, tmp_path) -> None:
        out = tmp_path / "report.html"
        html = render_backtest_report(_make_result(), out_path=out)
        assert out.exists()
        contents = out.read_text()
        assert contents == html

    def test_metadata_embedded(self) -> None:
        meta = {"ticker": "AAPL", "horizon_y": 1.0, "model": "vassalou_xing"}
        html = render_backtest_report(_make_result(), metadata=meta)
        for k, v in meta.items():
            assert str(k) in html
            assert str(v) in html
