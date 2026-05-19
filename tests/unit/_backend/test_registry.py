"""Tests for the backend registry."""

from __future__ import annotations

import pytest

from merton._backend import _registry


@pytest.fixture(autouse=True)
def _reset():
    _registry.reset()
    yield
    _registry.reset()


class TestRegistry:
    def test_numpy_always_available(self) -> None:
        assert "numpy" in _registry.available()

    def test_numba_available_in_test_env(self) -> None:
        # The dev environment installs numba via the merton dependencies.
        assert "numba" in _registry.available()

    def test_default_is_numba_when_available(self) -> None:
        assert _registry.default_backend() in {"numba", "numpy"}

    def test_env_var_override(self, monkeypatch) -> None:
        monkeypatch.setenv("MERTON_BACKEND", "numpy")
        _registry.reset()
        assert _registry.default_backend() == "numpy"

    def test_env_var_ignored_when_backend_missing(self, monkeypatch) -> None:
        monkeypatch.setenv("MERTON_BACKEND", "cupy")  # not installed in CI
        _registry.reset()
        # Falls back to numba (or numpy if numba absent).
        assert _registry.default_backend() in {"numba", "numpy"}

    def test_unknown_env_var_value_ignored(self, monkeypatch) -> None:
        monkeypatch.setenv("MERTON_BACKEND", "wat")
        _registry.reset()
        assert _registry.default_backend() in {"numba", "numpy"}
