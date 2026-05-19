"""Tests for the on-disk cache helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from merton import cache


@pytest.fixture(autouse=True)
def _reset_cache_state():
    """Make sure each test starts with caching disabled."""
    cache.disable()
    yield
    cache.disable()


class TestCache:
    def test_disabled_by_default(self) -> None:
        assert cache.is_enabled() is False

    def test_cached_is_passthrough_when_disabled(self) -> None:
        calls = {"n": 0}

        @cache.cached
        def f(x: int) -> int:
            calls["n"] += 1
            return x * 2

        assert f(3) == 6
        assert f(3) == 6
        assert calls["n"] == 2  # no caching when disabled

    def test_enabled_caches(self, tmp_path: Path) -> None:
        cache.enable(cache_dir=tmp_path)
        assert cache.is_enabled() is True
        calls = {"n": 0}

        @cache.cached
        def slow(x: int) -> int:
            calls["n"] += 1
            return x + 100

        assert slow(7) == 107
        assert slow(7) == 107
        # joblib caches by argument hash; the second call should be free.
        assert calls["n"] == 1

    def test_clear(self, tmp_path: Path) -> None:
        cache.enable(cache_dir=tmp_path)

        @cache.cached
        def g(x: int) -> int:
            return x * 3

        g(2)
        cache.clear()  # should not raise

    def test_default_cache_dir_used(self) -> None:
        """Calling enable() with no args still works (uses the platform default)."""
        cache.enable()
        assert cache.is_enabled() is True
        cache.disable()
