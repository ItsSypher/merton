"""Tests for the classic-xlwings UDF module.

We don't actually run a UDF in-process (that would require a live Excel
instance), but we verify the registration works once xlwings is on the
path and gracefully degrades when it isn't.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest


@pytest.fixture
def _udf_fresh():
    """Pop ``merton.excel.udf`` from ``sys.modules`` so each test re-imports it."""
    saved = sys.modules.pop("merton.excel.udf", None)
    yield
    if saved is not None:
        sys.modules["merton.excel.udf"] = saved
    else:
        sys.modules.pop("merton.excel.udf", None)


def test_udf_module_imports_with_xlwings_installed(_udf_fresh) -> None:
    """When xlwings is available, every EXCEL_FUNCTIONS name should be
    registered as a module-level attribute."""
    if importlib.util.find_spec("xlwings") is None:
        pytest.skip("xlwings not installed; tested via the degradation path")
    udf = importlib.import_module("merton.excel.udf")
    from merton.excel.functions import EXCEL_FUNCTIONS

    for name, _description, _fn in EXCEL_FUNCTIONS:
        assert hasattr(udf, name), f"{name} should be registered when xlwings is installed"


def test_udf_module_degrades_without_xlwings(monkeypatch, _udf_fresh) -> None:
    """When ``import xlwings`` fails, the registration returns ``{}``."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "xlwings":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    udf = importlib.import_module("merton.excel.udf")
    assert udf._register_with_xlwings() == {}
