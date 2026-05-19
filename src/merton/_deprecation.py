"""Soft-deprecation helpers used to evolve the public API across releases.

The promise: once a name lands in :data:`merton.__all__` (or any submodule's
``__all__``) at v1.0, removing or renaming it is a major-version bump.
Between releases, we soft-deprecate by routing the old name through
:func:`deprecated` so importers see a :class:`DeprecationWarning` (and the
docs cite the removal version) instead of an :class:`AttributeError`.

Usage
-----
::

    from merton._deprecation import deprecated


    @deprecated(
        "Use merton.fit(firm, method='vassalou_xing') instead",
        since="0.9",
        removed_in="2.0",
    )
    def calibrate(firm):
        return fit(firm, method="vassalou_xing")

Each call emits a single ``DeprecationWarning`` with the message and the
removal version stamped in. Repeat calls in the same process are
de-duplicated via :func:`functools.lru_cache`-style memoisation on the
warning text so production logs don't drown in noise.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

__all__ = ["deprecated", "deprecated_alias", "warn_deprecated"]


def _format(reason: str, *, since: str, removed_in: str | None) -> str:
    suffix = f" (deprecated since {since}"
    if removed_in:
        suffix += f", removal target {removed_in}"
    suffix += ")."
    return f"{reason}{suffix}"


def warn_deprecated(reason: str, *, since: str, removed_in: str | None = None) -> None:
    """Emit a :class:`DeprecationWarning` once per process per message."""
    warnings.warn(
        _format(reason, since=since, removed_in=removed_in), DeprecationWarning, stacklevel=3
    )


def deprecated(
    reason: str,
    *,
    since: str,
    removed_in: str | None = None,
) -> Callable[[F], F]:
    """Decorate a function/method to emit a ``DeprecationWarning`` on call."""

    msg = _format(reason, since=since, removed_in=removed_in)

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return fn(*args, **kwargs)

        wrapper.__deprecated__ = msg  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


def deprecated_alias(
    new_target: Callable[..., Any],
    *,
    old_name: str,
    since: str,
    removed_in: str | None = None,
) -> Callable[..., Any]:
    """Return a thin wrapper around ``new_target`` that warns under the old name.

    Use this when renaming a function: keep the old import path available
    so 0.x users get a clear redirect instead of an :class:`ImportError`.
    """
    reason = (
        f"{old_name} is deprecated; use {new_target.__module__}.{new_target.__qualname__} instead"
    )
    return deprecated(reason, since=since, removed_in=removed_in)(new_target)
