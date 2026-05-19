"""Tests for the merton._deprecation soft-deprecation helpers."""

from __future__ import annotations

import warnings

import pytest

from merton._deprecation import deprecated, deprecated_alias, warn_deprecated


class TestDeprecatedDecorator:
    def test_emits_deprecation_warning(self) -> None:
        @deprecated("Use new_thing", since="0.9", removed_in="2.0")
        def old_thing(x: int) -> int:
            return x + 1

        with pytest.warns(DeprecationWarning, match=r"Use new_thing.*0\.9.*2\.0"):
            assert old_thing(41) == 42

    def test_preserves_function_metadata(self) -> None:
        @deprecated("...", since="0.9")
        def f(x: int) -> int:
            """docstring"""
            return x

        assert f.__name__ == "f"
        assert f.__doc__ == "docstring"
        assert hasattr(f, "__deprecated__")

    def test_message_omits_removal_when_unspecified(self) -> None:
        @deprecated("Use bar", since="0.9")
        def foo() -> int:
            return 1

        with pytest.warns(DeprecationWarning) as records:
            foo()
        msg = str(records[0].message)
        assert "removal target" not in msg
        assert "0.9" in msg

    def test_warning_stacklevel_points_at_caller(self) -> None:
        @deprecated("Use bar", since="0.9")
        def foo() -> int:
            return 1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            foo()
        assert len(caught) == 1
        # The warning's stacklevel=2 means the filename should be this test file.
        assert caught[0].filename.endswith("test_deprecation.py")


class TestDeprecatedAlias:
    def test_alias_warns_and_forwards(self) -> None:
        def new_fn(x: int) -> int:
            return x * 2

        old_fn = deprecated_alias(new_fn, old_name="merton.old_fn", since="0.9", removed_in="2.0")
        with pytest.warns(DeprecationWarning, match="merton.old_fn"):
            assert old_fn(21) == 42


class TestWarnDeprecated:
    def test_emits_with_full_message(self) -> None:
        with pytest.warns(DeprecationWarning, match=r"foo.*0\.9"):
            warn_deprecated("foo deprecated", since="0.9")

    def test_includes_removal_target(self) -> None:
        with pytest.warns(DeprecationWarning) as records:
            warn_deprecated("foo deprecated", since="0.9", removed_in="2.0")
        assert "2.0" in str(records[0].message)
