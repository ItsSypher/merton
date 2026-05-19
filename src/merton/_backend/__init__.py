"""Array-backend dispatch.

A single :func:`resolve` helper figures out which backend to send a call to,
given the user's explicit preference, the namespace of input arrays, and
which backends are installed. All public numerical functions in :mod:`merton`
funnel through here.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from ..exceptions import BackendNotAvailableError, MertonBackendFallbackWarning
from ._registry import BackendName, available, default_backend

if TYPE_CHECKING:
    from collections.abc import Iterable


def _array_namespace_name(*arrays: Any) -> BackendName | None:
    """Best-effort detection of which backend an input array lives in."""
    for arr in arrays:
        module = type(arr).__module__
        if module.startswith("cupy"):
            return "cupy"
        if module.startswith("jax") or module.startswith("jaxlib"):
            return "jax"
        if module.startswith("mlx"):
            return "mlx"
    return None


def resolve(
    *arrays: Any,
    backend: str | None = None,
    size: int | None = None,
    numba_threshold: int = 256,
) -> BackendName:
    """Choose the backend for a numerical call.

    Resolution order (first match wins):

    1. Explicit ``backend=`` kwarg.
    2. Input-array namespace (``cupy.ndarray`` → ``cupy``, etc.).
    3. ``MERTON_BACKEND`` env var (via :func:`default_backend`).
    4. Size heuristic: arrays smaller than ``numba_threshold`` stay on NumPy;
       larger ones jump to Numba.

    If the chosen backend isn't installed, fall back to the next available
    one and emit :class:`MertonBackendFallbackWarning`.
    """
    avail = set(available())

    explicit: BackendName | None
    if backend and backend.lower() != "auto":
        explicit = backend.lower()  # type: ignore[assignment]
        if explicit not in {"numpy", "numba", "cupy", "jax", "mlx"}:
            raise BackendNotAvailableError(
                f"Unknown backend '{backend}'",
                suggested_fix="Choose one of: numpy, numba, cupy, jax, mlx.",
            )
        if explicit in avail:
            return explicit
        fallback = _fallback_for(explicit, avail)
        warnings.warn(
            f"Backend '{explicit}' is not installed; falling back to '{fallback}'.",
            MertonBackendFallbackWarning,
            stacklevel=3,
        )
        return fallback

    detected = _array_namespace_name(*arrays)
    if detected is not None:
        if detected in avail:
            return detected
        fallback = _fallback_for(detected, avail)
        warnings.warn(
            f"Detected '{detected}' arrays but that backend isn't installed; "
            f"falling back to '{fallback}'.",
            MertonBackendFallbackWarning,
            stacklevel=3,
        )
        return fallback

    chosen = default_backend()
    if chosen == "numba" and size is not None and size < numba_threshold:
        return "numpy"
    return chosen


def _fallback_for(requested: BackendName, avail: set[BackendName]) -> BackendName:
    """Pick a sensible substitute if ``requested`` isn't installed."""
    preference: dict[BackendName, Iterable[BackendName]] = {
        "cupy": ("numba", "numpy"),
        "jax": ("numba", "numpy"),
        "mlx": ("numba", "numpy"),
        "numba": ("numpy",),
        "numpy": ("numpy",),
    }
    for candidate in preference.get(requested, ("numpy",)):
        if candidate in avail:
            return candidate
    return "numpy"


def to_numpy(arr: Any) -> np.ndarray:
    """Return ``arr`` as a NumPy array regardless of source backend."""
    module = type(arr).__module__
    if module.startswith("cupy"):
        return arr.get()  # type: ignore[no-any-return]
    if module.startswith("jax"):
        return np.asarray(arr)
    if module.startswith("mlx"):
        return np.asarray(arr)
    return np.asarray(arr)


def get_kernel(backend: BackendName, name: str) -> Any:
    """Look up a kernel function for ``backend`` by its module-level ``name``."""
    if backend == "numpy":
        from . import _numpy as mod
    elif backend == "numba":
        from . import _numba as mod  # type: ignore[no-redef]
    elif backend == "cupy":
        from . import _cupy as mod  # type: ignore[no-redef]
    elif backend == "jax":
        from . import _jax as mod  # type: ignore[no-redef]
    elif backend == "mlx":
        from . import _mlx as mod  # type: ignore[no-redef]
    else:  # pragma: no cover
        raise BackendNotAvailableError(f"Unknown backend {backend!r}")
    fn = getattr(mod, name, None)
    if fn is None:
        raise BackendNotAvailableError(
            f"Backend {backend!r} has no kernel named {name!r}",
            suggested_fix="This operation may only be supported on a subset of backends.",
        )
    return fn


__all__ = ["available", "default_backend", "get_kernel", "resolve", "to_numpy"]
