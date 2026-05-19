"""Classic xlwings UDFs (Excel desktop, Windows-only fallback).

This module is imported by xlwings's ``RunPython`` when the user prefers
the legacy UDF pattern instead of the Office.js server. It registers
every :data:`merton.excel.functions.EXCEL_FUNCTIONS` entry as an
``@xw.func`` and re-exposes their docstrings via ``@xw.arg``.

Activation
----------
On Windows::

    xlwings addin install
    merton excel install --classic    # configures Excel to point at this module

This installs an XLSX add-in whose RUNPYTHON command imports
``merton.excel.udf``. The UDFs then become available in the workbook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .functions import ARG_DOCS, EXCEL_FUNCTIONS

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


def _register_with_xlwings() -> dict[str, Any]:
    """Re-decorate the functions with ``@xw.func`` / ``@xw.arg``.

    Returns a dict ``{excel_name: decorated_callable}``. Skipped when
    xlwings isn't installed — the function returns an empty dict and the
    caller can fall back to the server path.
    """
    try:
        import xlwings as xw  # type: ignore[import-not-found]
    except ImportError:
        return {}

    decorated: dict[str, Any] = {}
    for excel_name, _description, fn in EXCEL_FUNCTIONS:
        func_decorated = xw.func(fn)
        # The xlwings @func decorator wraps the callable but preserves the
        # signature; layer @xw.arg over each parameter for the help text.
        docs = ARG_DOCS.get(fn.__name__, {})
        for arg_name, arg_doc in docs.items():
            func_decorated = xw.arg(arg_name, doc=arg_doc)(func_decorated)
        # The wrapper's __name__ drives the Excel formula name.
        func_decorated.__name__ = excel_name  # type: ignore[attr-defined]
        decorated[excel_name] = func_decorated
    return decorated


# Populate module-level names so xlwings's UDF importer can find them.
_DECORATED = _register_with_xlwings()
for _name, _callable in _DECORATED.items():
    globals()[_name] = _callable

__all__ = list(_DECORATED)
