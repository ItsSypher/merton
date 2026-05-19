"""Tests for the ROCCurve and calibration_curve helpers."""

from __future__ import annotations

import numpy as np
import pytest

from merton.backtest import (
    Backtest,
    CalibrationCurve,
    ROCCurve,
    calibration_curve,
    roc_curve,
)
from merton.backtest.calibration import calibration_plot
from merton.exceptions import MertonInputError


class TestROCCurve:
    def test_returns_arrays_aligned(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.3, 100)
        s = rng.uniform(0, 1, 100)
        roc = roc_curve(s, y)
        assert isinstance(roc, ROCCurve)
        assert roc.fpr.shape == roc.tpr.shape

    def test_trapz_auc_matches_metric(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.binomial(1, 0.3, 500).astype(float)
        s = np.clip(y * 0.4 + rng.normal(0, 0.1, 500), 0, 1)
        from merton.backtest import auc as metric_auc

        roc = roc_curve(s, y)
        assert roc.auc() == pytest.approx(metric_auc(s, y), abs=1e-3)

    def test_undefined_one_class(self) -> None:
        with pytest.raises(MertonInputError):
            roc_curve(np.array([0.1, 0.2]), np.array([0, 0]))


class TestCalibrationCurve:
    def test_quantile_bins(self) -> None:
        rng = np.random.default_rng(2)
        s = rng.uniform(0, 1, 1_000)
        y = (rng.uniform(0, 1, 1_000) < s).astype(float)  # well-calibrated
        cc = calibration_curve(s, y, bins=10, strategy="quantile")
        assert isinstance(cc, CalibrationCurve)
        assert cc.mean_predicted.shape == (10,)
        # Well-calibrated model: fraction_positives ≈ mean_predicted.
        gap = np.abs(cc.mean_predicted - cc.fraction_positives)
        assert np.nanmax(gap) < 0.1

    def test_uniform_bins(self) -> None:
        rng = np.random.default_rng(3)
        s = rng.uniform(0, 1, 1_000)
        y = (rng.uniform(0, 1, 1_000) < 0.3).astype(float)
        cc = calibration_curve(s, y, bins=5, strategy="uniform")
        assert cc.bin_counts.sum() == 1_000

    def test_unknown_strategy(self) -> None:
        with pytest.raises(MertonInputError):
            calibration_curve(np.array([0.1, 0.5]), np.array([0, 1]), strategy="bogus")


class TestBacktestOrchestrator:
    def test_run_with_arrays(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.3, 1_000).astype(float)
        s = np.clip(y * 0.3 + rng.normal(0, 0.1, 1_000), 0, 1)
        result = Backtest().run(s, y, n_bins=10)
        assert 0 < result.auc <= 1
        assert 0 <= result.brier <= 1
        text = result.summary()
        assert "AUC" in text
        d = result.to_dict()
        assert "auc" in d

    def test_extra_metrics(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.3, 200).astype(float)
        s = np.clip(rng.uniform(0, 1, 200), 0, 1)

        def my_mean_pd(p, _y):
            return float(p.mean())

        bt = Backtest().add_metric("mean_pd", my_mean_pd)
        result = bt.run(s, y)
        assert "mean_pd" in result.extra

    def test_run_from_panel(self) -> None:
        import pandas as pd

        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "default": rng.binomial(1, 0.2, 200),
                "pd": np.clip(rng.uniform(0, 1, 200), 0, 1),
            }
        )
        bt = Backtest(df)
        result = bt.run()
        assert 0 < result.auc < 1

    def test_run_without_panel_or_arrays_raises(self) -> None:
        with pytest.raises(MertonInputError):
            Backtest().run()


class TestCalibrationPlot:
    def test_plot_runs(self) -> None:
        pytest.importorskip("matplotlib")
        rng = np.random.default_rng(0)
        s = rng.uniform(0, 1, 200)
        y = (rng.uniform(0, 1, 200) < s).astype(float)
        ax = calibration_plot(s, y, bins=5)
        assert ax is not None
