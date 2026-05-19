"""FastAPI app that hosts the merton Excel custom functions.

The Office.js custom-function protocol expects two JSON documents and a
JavaScript runtime:

- ``GET /functions.json`` — metadata describing each function, its
  parameters, and return type.
- ``GET /static/functions.html`` — a tiny HTML page that loads
  ``functions.js``.
- ``GET /static/functions.js`` — calls ``CustomFunctions.associate(...)``
  for each function name, forwarding invocations to our ``POST /call``
  endpoint.
- ``POST /call`` — receives ``{"function": str, "args": list}``, executes
  the matching Python function, and returns ``{"result": ...}``.

This module is intentionally framework-light: we don't depend on
``xlwings-server`` to keep the dependency footprint small. Anyone who
prefers the xlwings-server route can still import
:mod:`merton.excel.functions` and register the callables manually.
"""

from __future__ import annotations

import inspect
from typing import Any

from .functions import ARG_DOCS, EXCEL_FUNCTIONS


def _params_for(fn) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    sig = inspect.signature(fn)
    name = fn.__name__
    docs = ARG_DOCS.get(name, {})
    params: list[dict[str, Any]] = []
    for p in sig.parameters.values():
        is_array = p.name in {"horizons", "pd_predictions", "defaults"}
        is_optional = p.default is not inspect.Parameter.empty
        params.append(
            {
                "name": p.name,
                "description": docs.get(p.name, ""),
                "type": "matrix" if is_array else "any",
                "dimensionality": "matrix" if is_array else "scalar",
                "optional": is_optional,
            }
        )
    return params


def build_functions_metadata() -> dict[str, Any]:
    """Construct the ``functions.json`` payload Office.js expects."""
    functions = []
    for excel_name, description, fn in EXCEL_FUNCTIONS:
        # Office.js expects ``id`` and ``name`` separately; ``name`` is what
        # appears in the cell, ``id`` is the JS associate() key.
        short = excel_name.removeprefix("MERTON_")
        functions.append(
            {
                "id": short,
                "name": short,
                "description": description,
                "helpUrl": "https://merton.readthedocs.io/excel/functions",
                "parameters": _params_for(fn),
                "result": {"type": "any", "dimensionality": "matrix"},
            }
        )
    return {"functions": functions}


def _functions_js(namespace: str = "MERTON") -> str:
    short_names = [name.removeprefix("MERTON_") for name, _, _ in EXCEL_FUNCTIONS]
    associates = "\n  ".join(
        f'CustomFunctions.associate("{n}", makeProxy("{n}"));' for n in short_names
    )
    return f"""// merton Excel custom functions — auto-generated.
// Each function forwards arguments to the local merton server.

function makeProxy(name) {{
  return async function(...args) {{
    const response = await fetch("/call", {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify({{function: name, args}}),
    }});
    if (!response.ok) {{
      throw new CustomFunctions.Error(
        CustomFunctions.ErrorCode.invalidValue,
        await response.text()
      );
    }}
    const payload = await response.json();
    if (payload.error) {{
      throw new CustomFunctions.Error(
        CustomFunctions.ErrorCode.invalidValue, payload.error
      );
    }}
    return payload.result;
  }};
}}

{associates}
"""


def _functions_html() -> str:
    return (
        "<!doctype html><html><head>"
        '<meta charset="utf-8"><title>merton</title>'
        '<script src="https://appsforoffice.microsoft.com/lib/1/hosted/office.js"></script>'
        '<script src="/static/functions.js"></script>'
        "</head><body></body></html>"
    )


def _taskpane_html() -> str:
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>merton</title></head>'
        "<body><h1>merton</h1>"
        "<p>Custom functions are registered under the <code>MERTON.*</code> "
        "namespace. Try <code>=MERTON.DD(100, 0.30, 35, 0.04, 1)</code>.</p>"
        '<p>See <a href="https://merton.readthedocs.io/excel/functions">the docs</a> '
        "for the complete reference.</p>"
        "</body></html>"
    )


try:
    from pydantic import BaseModel as _BaseModel

    class CallPayload(_BaseModel):
        function: str
        args: list[Any]

except ImportError:  # pragma: no cover - pydantic always installed via core deps
    CallPayload = None  # type: ignore[assignment, misc]


def make_app(namespace: str = "MERTON"):  # type: ignore[no-untyped-def]
    """Build and return the FastAPI app.

    Lazy-imports FastAPI so that ``import merton.excel`` does not pull in
    the heavy web stack unless the user is actively starting the server.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
    except ImportError as err:  # pragma: no cover - optional dep
        raise ImportError(
            'merton.excel.server requires the excel extra: pip install "merton[excel]"'
        ) from err

    app = FastAPI(title="merton Excel server", version="0.6.0")
    registry = {name.removeprefix("MERTON_"): fn for name, _, fn in EXCEL_FUNCTIONS}

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "service": "merton"}

    @app.get("/functions.json")
    async def functions_json() -> JSONResponse:
        return JSONResponse(build_functions_metadata())

    @app.get("/static/functions.js")
    async def functions_js() -> PlainTextResponse:
        return PlainTextResponse(_functions_js(namespace), media_type="application/javascript")

    @app.get("/static/functions.html")
    async def functions_html() -> HTMLResponse:
        return HTMLResponse(_functions_html())

    @app.get("/taskpane.html")
    async def taskpane() -> HTMLResponse:
        return HTMLResponse(_taskpane_html())

    @app.post("/call")
    async def call(payload: CallPayload) -> dict[str, Any]:
        fn = registry.get(payload.function)
        if fn is None:
            raise HTTPException(status_code=404, detail=f"unknown function {payload.function!r}")
        try:
            result = fn(*payload.args)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"result": result}

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000, *, reload: bool = False) -> None:
    """Start the FastAPI server with uvicorn.

    Used by the ``merton excel server start`` CLI command.
    """
    try:
        import uvicorn
    except ImportError as err:  # pragma: no cover
        raise ImportError("uvicorn is required: install merton[excel]") from err
    uvicorn.run(
        "merton.excel.server:_make_app_for_uvicorn",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


def _make_app_for_uvicorn():  # type: ignore[no-untyped-def]
    """uvicorn-friendly factory (avoids module-level FastAPI import)."""
    return make_app()


__all__ = ["build_functions_metadata", "make_app", "run_server"]
