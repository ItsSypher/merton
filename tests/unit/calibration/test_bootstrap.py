"""Tests for the block-bootstrap utility."""

from __future__ import annotations

import numpy as np

from merton import Firm, MertonModel
from merton.calibration import block_bootstrap_calibration


def _series() -> np.ndarray:
    rng = np.random.default_rng(11)
    log_ret = rng.normal(0.0003, 0.02, size=252)
    return 100.0 * np.exp(np.cumsum(log_ret))


class TestBootstrap:
    def test_dict_refit(self) -> None:
        eq = _series()

        def refit(sample):
            # Pretend calibration: just return summary stats.
            r = np.diff(np.log(sample))
            return {
                "asset_vol": float(r.std(ddof=1) * np.sqrt(252)),
                "mean_return": float(r.mean() * 252),
            }

        bs = block_bootstrap_calibration(eq, refit=refit, n_resamples=50, block_length=20, seed=1)
        assert bs.n_resamples == 50
        assert "asset_vol" in bs.parameter_samples
        ci = bs.ci("asset_vol", level=0.90)
        assert ci.lower < ci.upper

    def test_derived_quantities(self) -> None:
        eq = _series()

        def refit(sample):
            r = np.diff(np.log(sample))
            return {"sigma": float(r.std(ddof=1) * np.sqrt(252))}

        bs = block_bootstrap_calibration(
            eq,
            refit=refit,
            n_resamples=40,
            seed=2,
            derived={"sigma_squared": lambda p: p["sigma"] ** 2},
        )
        assert "sigma_squared" in bs.derived_samples
        ci = bs.ci("sigma_squared", level=0.95)
        assert ci.lower >= 0


class TestModelBootstrap:
    def test_model_bootstrap_via_vassalou_xing(self) -> None:
        eq = _series()
        firm = Firm(equity=eq, debt_short=20, debt_long=30, rf=0.04, horizon=1.0)
        result = MertonModel(method="vassalou_xing", n_bootstrap=30, random_state=3).fit(firm)
        ci = result.confidence_interval(level=0.95, method="bootstrap")
        assert "asset_vol" in ci
        assert "dd" in ci
        assert ci["asset_vol"].method == "bootstrap"
