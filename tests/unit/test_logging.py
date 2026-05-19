"""Tests for the structlog wiring helper."""

from __future__ import annotations

import logging

from merton.logging import enable, get_logger


class TestLogging:
    def test_get_logger_returns_bound_logger(self) -> None:
        log = get_logger("merton.test")
        assert log is not None
        # structlog bound loggers expose info/warning/error methods.
        assert hasattr(log, "info")

    def test_enable_with_string_level(self) -> None:
        enable("DEBUG")
        # No exception means the wiring worked.

    def test_enable_with_int_level(self) -> None:
        enable(logging.WARNING)

    def test_log_call_does_not_raise(self) -> None:
        log = get_logger("merton.test")
        log.info("hello", foo=1)
        log.warning("careful", bar=2)
