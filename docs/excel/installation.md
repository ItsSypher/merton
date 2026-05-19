# Excel — installation and sideloading

The merton Excel add-in is **two pieces**:

1. A small FastAPI server that runs locally and exposes the
   `=MERTON.*` formulas.
2. An Office.js manifest XML that tells Excel where to find the server.

## Install the extra

```{code-block} bash
uv pip install "merton[excel]"
```

This pulls in `xlwings`, `fastapi`, `uvicorn`, `openpyxl`, `xlsxwriter`,
`jinja2`, and `httpx`.

## Sideload the add-in

```{code-block} bash
merton excel install --url http://localhost:8000
```

This writes `merton-manifest.xml` into your Excel sideload directory:

- macOS — `~/Library/Containers/com.Microsoft.Excel/Data/Documents/wef/`
- Windows — `%LOCALAPPDATA%\Microsoft\Office\16.0\Wef\`
- Linux / others — `~/.config/merton/excel/` (manual upload via Excel
  on the web)

To verify:

```{code-block} bash
merton excel status
```

## Start the server

```{code-block} bash
merton excel server start --port 8000
```

Leave the terminal running while you use Excel. To run it detached:

```{code-block} bash
merton excel server start --background
merton excel server stop
```

## Activate the add-in in Excel

1. Open Excel (desktop or web).
2. **Insert → My Add-ins → Shared Folder** (or *Upload My Add-in* on the
   web).
3. Pick **merton**.
4. Try `=MERTON.DD(100, 0.30, 35, 0.04, 1)`.

Excel will fetch the function metadata, register the namespace, and
autocomplete every `MERTON.*` formula.

## Uninstall

```{code-block} bash
merton excel server stop
merton excel uninstall
```

## Classic UDF fallback (Windows desktop only)

If you can't run a local server (e.g. inside a locked-down workstation),
the same formulas register as classic xlwings UDFs via
`merton.excel.udf`. Configure `xlwings` to point at that module, then
the formulas are available without the FastAPI server. See
{doc}`/cookbook/excel-dashboard` for the full setup.
