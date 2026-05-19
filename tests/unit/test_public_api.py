"""Public-API contract tests.

These tests are the safety net for the v1.0 API freeze: anything in
:data:`merton.__all__` must remain importable, callable on at least a
trivial input, and stay alphabetically sorted (for cheap conflict-free
diffs across PRs).
"""

from __future__ import annotations

import merton


def test_all_names_resolve() -> None:
    """Every name in merton.__all__ must resolve to something."""
    for name in merton.__all__:
        attr = getattr(merton, name)
        assert attr is not None, f"merton.{name} resolved to None"


def test_all_is_sorted() -> None:
    """`__all__` must be sorted to keep diffs minimal and to make the
    public surface easy to scan."""
    assert merton.__all__ == sorted(merton.__all__), (
        "merton.__all__ is unsorted — keep it alphabetical so PR diffs stay tight."
    )


def test_no_private_names_in_all() -> None:
    """No leading-underscore names should leak into the public surface."""
    leaks = [n for n in merton.__all__ if n.startswith("_") and n != "__version__"]
    assert not leaks, f"private names leaked into merton.__all__: {leaks}"


def test_version_is_pep440() -> None:
    """`merton.__version__` must be a PEP 440 string."""
    import re

    pep440 = re.compile(r"^\d+\.\d+\.\d+(?:[ab]\d+|rc\d+|\.dev\d+)?(?:\+[\w.]+)?$")
    assert pep440.match(merton.__version__), f"Bad version: {merton.__version__!r}"


def test_lazy_submodules_load() -> None:
    """Touching a lazy submodule should resolve it via __getattr__."""
    for sub in ("scenarios", "obs", "excel", "cli"):
        mod = getattr(merton, sub)
        assert mod.__name__.startswith("merton.")


def test_top_level_namespaces_have_all() -> None:
    """Each public submodule should declare its own `__all__`."""
    namespaces = (
        merton.calibration,
        merton.backtest,
        merton.extensions,
        merton.greeks,
        merton.portfolio,
        merton.reports,
        merton.scenarios,
        merton.obs,
        merton.exceptions,
    )
    for ns in namespaces:
        assert hasattr(ns, "__all__"), f"{ns.__name__} has no __all__"
        assert isinstance(ns.__all__, list), f"{ns.__name__}.__all__ is not a list"
        assert ns.__all__, f"{ns.__name__}.__all__ is empty"
