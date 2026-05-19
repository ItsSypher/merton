"""Structured logging wiring for merton.

The library uses :mod:`structlog`. By default it emits at ``WARNING`` and writes
human-readable output to stderr when run interactively, JSON otherwise.
Applications can configure logging themselves; we never touch the stdlib root
logger except via the user-facing :func:`enable` helper.
"""

from __future__ import annotations

import logging
import sys

import structlog

from ._config import get_config


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger named ``name`` (or the module's name)."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def enable(level: str | int = "INFO") -> None:
    """Wire up a sensible default logging configuration.

    Convenience for interactive use; libraries should not call this from their
    own ``__init__`` modules.
    """
    if isinstance(level, str):
        level = level.upper()
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)
    _configure_structlog(level)


def _configure_structlog(level: str | int) -> None:
    is_tty = sys.stderr.isatty()
    renderer: structlog.types.Processor = (
        structlog.dev.ConsoleRenderer() if is_tty else structlog.processors.JSONRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level) if isinstance(level, str) else level
        ),
        cache_logger_on_first_use=True,
    )


_configure_structlog(get_config().log_level)


__all__ = ["enable", "get_logger"]
