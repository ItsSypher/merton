"""Lazy detection of which backends are installed in the current interpreter."""

from __future__ import annotations

import importlib.util
import os
from functools import cache
from typing import Literal

BackendName = Literal["numpy", "numba", "cupy", "jax", "mlx"]


@cache
def _has(module: str) -> bool:
    """Cheap check whether ``module`` can be imported without doing so."""
    return importlib.util.find_spec(module) is not None


@cache
def available() -> tuple[BackendName, ...]:
    """Return the tuple of backends importable in this interpreter.

    ``numpy`` is always present (mandatory dependency). ``numba`` is also a
    hard dep, but we still gate on import so a user who deliberately removes
    it gets a graceful degradation.
    """
    backends: list[BackendName] = ["numpy"]
    if _has("numba"):
        backends.append("numba")
    if _has("cupy"):
        backends.append("cupy")
    if _has("jax"):
        backends.append("jax")
    if _has("mlx"):
        backends.append("mlx")
    return tuple(backends)


@cache
def default_backend() -> BackendName:
    """Resolve the backend that ``backend="auto"`` should map to.

    Order:

    1. ``MERTON_BACKEND`` env var, if set to one of the installed backends.
    2. ``numba`` if available.
    3. ``numpy``.
    """
    env = os.environ.get("MERTON_BACKEND", "").lower()
    avail = set(available())
    if env in {"numpy", "numba", "cupy", "jax", "mlx"} and env in avail:
        return env  # type: ignore[return-value]
    if "numba" in avail:
        return "numba"
    return "numpy"


def reset() -> None:
    """Clear the lookup caches (test-only helper)."""
    _has.cache_clear()
    available.cache_clear()
    default_backend.cache_clear()


__all__ = ["BackendName", "available", "default_backend", "reset"]
