"""Tests for the merton exception hierarchy and suggested_fix behaviour."""

from __future__ import annotations

import pytest

from merton import exceptions as e


class TestExceptionHierarchy:
    def test_input_error_is_value_error(self) -> None:
        assert issubclass(e.MertonInputError, ValueError)
        assert issubclass(e.MertonInputError, e.MertonError)

    def test_dimension_mismatch_is_input_error(self) -> None:
        assert issubclass(e.DimensionMismatchError, e.MertonInputError)

    def test_calibration_convergence_is_calibration_error(self) -> None:
        assert issubclass(e.CalibrationConvergenceError, e.CalibrationError)

    def test_backend_not_available_is_backend_error(self) -> None:
        assert issubclass(e.BackendNotAvailableError, e.BackendError)


class TestSuggestedFix:
    def test_message_includes_fix_when_provided(self) -> None:
        with pytest.raises(e.MertonError) as excinfo:
            raise e.MertonError("something broke", suggested_fix="try X")
        text = str(excinfo.value)
        assert "something broke" in text
        assert "try X" in text

    def test_message_clean_without_fix(self) -> None:
        with pytest.raises(e.MertonError) as excinfo:
            raise e.MertonError("only message")
        text = str(excinfo.value)
        assert "only message" in text
        assert "Suggested fix" not in text

    def test_input_error_accepts_kwarg_via_mro(self) -> None:
        """The MRO fix from Phase 0.1 — ValueError-flavoured subclasses
        must still accept the suggested_fix keyword arg."""
        with pytest.raises(e.MertonInputError) as excinfo:
            raise e.MertonInputError("bad input", suggested_fix="hint")
        assert "hint" in str(excinfo.value)


class TestWarnings:
    def test_backend_fallback_warning_is_user_warning(self) -> None:
        assert issubclass(e.MertonBackendFallbackWarning, UserWarning)

    def test_numerical_warning_is_user_warning(self) -> None:
        assert issubclass(e.MertonNumericalWarning, UserWarning)
