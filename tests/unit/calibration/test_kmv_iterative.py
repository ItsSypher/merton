"""Tests for the KMV iterative calibrator."""

from __future__ import annotations

import numpy as np
import pytest

from merton import Firm, MertonModel
from merton.calibration import kmv_iterative


class TestKMVIterative:
    def test_basic_fit(self, simple_firm: Firm) -> None:
        res = kmv_iterative(
            equity=float(simple_firm.equity),
            equity_vol=float(simple_firm.equity_vol),
            debt=float(simple_firm.default_point_value()),
            rf=float(simple_firm.rf),
            T=float(simple_firm.horizon),
        )
        assert res.converged
        assert res.method == "kmv_iterative"
        # Risk-neutral EDF in the diagnostics should match the eventual PD.
        assert "risk_neutral_edf" in res.diagnostics

    def test_matches_jmr_on_kmv_default_point(self, simple_firm: Firm) -> None:
        """When the default point is KMV, KMV iterative ≡ JMR iterative."""
        kmv = MertonModel(method="kmv_iterative").fit(simple_firm)
        jmr = MertonModel(method="jmr_iterative").fit(simple_firm)
        np.testing.assert_allclose(kmv.asset_vol, jmr.asset_vol, rtol=1e-5)
        np.testing.assert_allclose(float(kmv.dd), float(jmr.dd), rtol=1e-5)

    def test_custom_edf_map(self, simple_firm: Firm) -> None:
        """A user-supplied EDF map should override the placeholder."""

        # A toy mapping: floor at 1bp; otherwise PD = max(0, 0.01 - 0.001 * DD).
        def edf_map(dd: float) -> float:
            return max(0.0001, 0.01 - 0.001 * dd)

        res = kmv_iterative(
            equity=float(simple_firm.equity),
            equity_vol=float(simple_firm.equity_vol),
            debt=float(simple_firm.default_point_value()),
            rf=float(simple_firm.rf),
            T=float(simple_firm.horizon),
            edf_map=edf_map,
        )
        # The empirical EDF differs from the risk-neutral placeholder.
        assert res.diagnostics["uses_proprietary_edf_map"]

    def test_requires_equity_vol(self) -> None:
        from merton.exceptions import MertonInputError

        firm = Firm(equity=100, debt_short=20, debt_long=30)
        with pytest.raises(MertonInputError):
            MertonModel(method="kmv_iterative").fit(firm)
