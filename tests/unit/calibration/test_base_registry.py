"""Tests for the Calibrator ABC and the calibration registry."""

from __future__ import annotations

import pytest

from merton import Firm
from merton.calibration.base import (
    CalibrationResult,
    Calibrator,
    available_methods,
    get,
    register,
)
from merton.exceptions import MertonInputError


class TestRegistry:
    def test_get_unknown_method_raises(self) -> None:
        with pytest.raises(MertonInputError):
            get("does_not_exist")

    def test_register_decorator(self) -> None:
        @register("toy_method")
        class _Toy(Calibrator):
            method = "toy_method"

            def fit(self, firm: Firm) -> CalibrationResult:
                return CalibrationResult(
                    asset_value=firm.equity + firm.total_debt,
                    asset_vol=0.1,
                    method="toy_method",
                )

        assert "toy_method" in available_methods()
        cls = get("toy_method")
        assert cls is _Toy

    def test_calibration_result_defaults(self) -> None:
        res = CalibrationResult(asset_value=100.0, asset_vol=0.2)
        assert res.method == "unknown"
        assert res.converged is True
        assert res.diagnostics == {}
