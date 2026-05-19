"""Excel integration for merton.

Two integration paths share this module:

1. **xlwings Server** (web + macOS + Windows via M365): a FastAPI app
   served by ``merton excel server start`` that Office.js sideloads via a
   generated manifest. This is the recommended path because it works in
   Excel on the web.

2. **Classic xlwings UDFs** (Windows desktop only): the same Python
   functions registered with the legacy ``@xw.func`` decorator. Useful
   for offline workflows where running a local HTTP server isn't
   convenient.

Install with::

    pip install "merton[excel]"

then::

    merton excel install
    merton excel server start --port 8000

Loading this module does **not** require xlwings — the math wrappers are
plain Python. The server / manifest / installer paths only kick in once
the user runs ``merton excel ...`` (or imports
:mod:`merton.excel.server`).
"""

from __future__ import annotations

from .functions import (
    merton_asset_value,
    merton_asset_vol,
    merton_backtest,
    merton_black_cox,
    merton_dd,
    merton_greeks,
    merton_pd,
    merton_pd_term,
    merton_portfolio_var,
    merton_spread,
)

__all__ = [
    "merton_asset_value",
    "merton_asset_vol",
    "merton_backtest",
    "merton_black_cox",
    "merton_dd",
    "merton_greeks",
    "merton_pd",
    "merton_pd_term",
    "merton_portfolio_var",
    "merton_spread",
]
