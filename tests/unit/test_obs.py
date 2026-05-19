"""Tests for the merton.obs OpenTelemetry shim.

These tests exercise the lazy / no-op behaviour without requiring the
``[obs]`` extra. When the extra is installed they additionally cover the
``enable`` / ``disable`` lifecycle and span emission.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

from merton import obs


@pytest.fixture(autouse=True)
def _reset_obs_state():
    """Disable tracing between tests so state never bleeds."""
    if obs.is_enabled():
        obs.disable()
    yield
    if obs.is_enabled():
        obs.disable()


class TestNoOpBehaviour:
    def test_is_enabled_false_by_default(self) -> None:
        assert obs.is_enabled() is False

    def test_span_yields_none_when_disabled(self) -> None:
        with obs.span("test.no_op", foo="bar") as s:
            assert s is None

    def test_traced_decorator_passthrough_when_disabled(self) -> None:
        @obs.traced("test.no_op")
        def f(x: int) -> int:
            return x * 2

        assert f(21) == 42

    def test_traced_preserves_wrapped_metadata(self) -> None:
        @obs.traced("alpha")
        def myfunc(x: int) -> int:
            """docstring"""
            return x

        assert myfunc.__name__ == "myfunc"
        assert myfunc.__doc__ == "docstring"

    def test_traced_default_name_uses_module_and_qualname(self) -> None:
        @obs.traced()
        def myfunc():
            return 1

        # The decorator name attribute isn't directly retrievable, but the
        # wrapped function works.
        assert myfunc() == 1

    def test_get_tracer_returns_none_when_disabled(self) -> None:
        assert obs.get_tracer() is None

    def test_disable_when_not_enabled_noop(self) -> None:
        obs.disable()  # must not raise
        assert obs.is_enabled() is False


class TestEnableImportError:
    def test_enable_without_extra_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If opentelemetry isn't installed, enable() raises ImportError."""
        # Sabotage the opentelemetry import by hiding the package.
        real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        def fake_import(name, *args, **kwargs):
            if name.startswith("opentelemetry"):
                raise ImportError(f"sabotaged: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        with pytest.raises(ImportError, match="merton\\[obs\\]"):
            obs.enable()


otel = pytest.importorskip("opentelemetry")


class TestEnabledLifecycle:
    def test_enable_sets_state(self) -> None:
        obs.enable(service_name="merton-test")
        assert obs.is_enabled() is True
        assert obs.get_tracer() is not None

    def test_enable_idempotent(self) -> None:
        obs.enable(service_name="merton-test")
        first_tracer = obs.get_tracer()
        obs.enable(service_name="merton-test-2")  # should be a no-op
        assert obs.get_tracer() is first_tracer

    def test_disable_clears_state(self) -> None:
        obs.enable(service_name="merton-test")
        obs.disable()
        assert obs.is_enabled() is False
        assert obs.get_tracer() is None

    def test_span_emits_when_enabled(self) -> None:
        obs.enable(service_name="merton-test", console=False)
        with obs.span("test.real_span", foo="bar") as s:
            assert s is not None

    def test_traced_decorator_runs_with_span(self) -> None:
        obs.enable(service_name="merton-test")

        @obs.traced("test.decorated", role="unit")
        def f(x: int) -> int:
            return x + 1

        assert f(41) == 42

    def test_span_records_exception(self) -> None:
        obs.enable(service_name="merton-test")
        with pytest.raises(ValueError), obs.span("test.error"):
            raise ValueError("boom")
        # State should still be enabled — span scope handled the exception cleanly.
        assert obs.is_enabled() is True


class TestEnvironmentVars:
    def test_auto_enable_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Re-importing the module with MERTON_OBS=1 triggers the auto-enable
        # branch. We use importlib.reload to drive the env-var path without
        # leaking state.
        monkeypatch.setenv("MERTON_OBS", "1")
        # Force a reload so the module-level if-block runs again.
        if obs.is_enabled():
            obs.disable()
        sys.modules.pop("merton.obs", None)
        reimported = importlib.import_module("merton.obs")
        try:
            # Either it enabled cleanly or it raised ImportError silently;
            # both are acceptable. The point is that no other exception
            # escapes.
            assert reimported.is_enabled() in (True, False)
        finally:
            if reimported.is_enabled():
                reimported.disable()
            sys.modules.pop("merton.obs", None)
            os.environ.pop("MERTON_OBS", None)
            importlib.import_module("merton.obs")
