"""Disk-backed memoization for expensive calibration outputs.

Wraps :class:`joblib.Memory` keyed on input array hashes. Off by default;
call :func:`enable` to opt in.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from joblib import Memory

from ._config import get_config

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeVar

    F = TypeVar("F", bound=Callable[..., object])


_memory: Memory | None = None
_enabled: bool = False


def _get_memory() -> Memory:
    global _memory
    if _memory is None:
        cache_dir: Path = get_config().cache_dir / "joblib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        _memory = Memory(location=str(cache_dir), verbose=0)
    return _memory


def enable(cache_dir: Path | str | None = None) -> None:
    """Turn on persistent calibration caching."""
    global _enabled, _memory
    if cache_dir is not None:
        _memory = Memory(location=str(cache_dir), verbose=0)
    _enabled = True


def disable() -> None:
    """Turn off persistent calibration caching."""
    global _enabled
    _enabled = False


def is_enabled() -> bool:
    return _enabled


def clear() -> None:
    """Drop the entire on-disk cache."""
    _get_memory().clear(warn=False)


def cached(func: F) -> F:
    """Decorator: cache the function's outputs on disk when caching is on.

    No-op (returns the function unchanged) when caching is disabled, so users
    pay zero overhead unless they opt in.
    """

    def wrapper(*args: object, **kwargs: object) -> object:
        if _enabled:
            return _get_memory().cache(func)(*args, **kwargs)
        return func(*args, **kwargs)

    wrapper.__wrapped__ = func  # type: ignore[attr-defined]
    wrapper.__name__ = getattr(func, "__name__", "cached")
    wrapper.__doc__ = func.__doc__
    return wrapper  # type: ignore[return-value]


__all__ = ["cached", "clear", "disable", "enable", "is_enabled"]
