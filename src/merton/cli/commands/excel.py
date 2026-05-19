"""``merton excel`` — install/uninstall the Office.js add-in and run the
local FastAPI server that powers it.

::

    merton excel install                  # write manifest to Excel sideload dir
    merton excel server start --port 8000 # foreground; ^C to stop
    merton excel server status            # check the PID file
    merton excel server stop              # graceful shutdown
    merton excel uninstall                # remove the manifest
    merton excel sample-workbook --out=sample.xlsx   # generate worked examples
"""

from __future__ import annotations

import os
import signal
import subprocess  # nosec B404 - used only to spawn the package-internal `merton excel server start` worker, never user input
import sys
from pathlib import Path

import typer
from platformdirs import user_runtime_dir
from rich.console import Console

from ...excel import installer

app = typer.Typer(no_args_is_help=True, help="Excel integration helpers.")
server_app = typer.Typer(no_args_is_help=True, help="Local FastAPI server controls.")
app.add_typer(server_app, name="server")
console = Console()


def _pid_path() -> Path:
    return Path(user_runtime_dir("merton")) / "excel-server.pid"


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------


@app.command("install")
def install(
    base_url: str = typer.Option(
        "http://localhost:8000",
        "--url",
        "-u",
        help="Where the FastAPI server will be reachable.",
    ),
) -> None:
    """Write the Office.js manifest into Excel's sideload directory."""
    path = installer.install(base_url=base_url)
    console.print(f"[green]Installed manifest[/]: {path}")
    console.print(
        "Next: open Excel, go to [bold]Insert → My Add-ins → Shared Folder[/], "
        "and pick [bold]merton[/]."
    )


@app.command("uninstall")
def uninstall() -> None:
    """Remove the merton manifest from Excel's sideload directory."""
    if installer.uninstall():
        console.print(f"[green]Removed manifest[/] from {installer.sideload_directory()}")
    else:
        console.print("[yellow]Manifest was not installed.[/]")


@app.command("status")
def status() -> None:
    """Print whether the manifest and server are currently active."""
    installed = installer.is_installed()
    sideload = installer.sideload_directory()
    console.print(f"Manifest installed : {'[green]yes[/]' if installed else '[red]no[/]'}")
    console.print(f"Sideload directory : [dim]{sideload}[/]")
    pid_file = _pid_path()
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # signal 0 = check liveness
            console.print(f"Server PID         : [green]{pid}[/]")
        except (OSError, ValueError):
            console.print("Server PID         : [yellow]stale (cleaning up)[/]")
            pid_file.unlink()
    else:
        console.print("Server PID         : [dim]not running[/]")


# ---------------------------------------------------------------------------
# server start / stop / status
# ---------------------------------------------------------------------------


@server_app.command("start")
def server_start(
    host: str = typer.Option("127.0.0.1", "--host", help="Host interface."),
    port: int = typer.Option(8000, "--port", "-p", help="TCP port."),
    reload: bool = typer.Option(False, "--reload", help="Restart on code changes."),
    background: bool = typer.Option(False, "--background", "-b", help="Run detached."),
) -> None:
    """Start the FastAPI server."""
    from ...excel.server import run_server

    if background:
        pid_file = _pid_path()
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(  # nosec B603 - all argv elements are package-controlled string literals, no shell, no user-supplied input
            [
                sys.executable,
                "-m",
                "merton",
                "excel",
                "server",
                "start",
                "--host",
                host,
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        pid_file.write_text(str(proc.pid))
        console.print(f"[green]Server started in background[/] (PID {proc.pid})")
        console.print(f"Visit http://{host}:{port}/healthz to confirm.")
        return
    console.print(f"[green]Starting merton Excel server on[/] http://{host}:{port}")
    console.print("Press [bold]Ctrl-C[/] to stop.")
    run_server(host=host, port=port, reload=reload)


@server_app.command("stop")
def server_stop() -> None:
    """Stop a background server started via ``server start --background``."""
    pid_file = _pid_path()
    if not pid_file.exists():
        console.print("[yellow]No PID file found; server is not running.[/]")
        return
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        console.print("[red]Corrupt PID file.[/]")
        pid_file.unlink()
        return
    try:
        os.kill(pid, signal.SIGTERM)
        console.print(f"[green]Sent SIGTERM[/] to PID {pid}.")
    except ProcessLookupError:
        console.print(f"[yellow]No process with PID {pid}; cleaning up.[/]")
    pid_file.unlink()


@server_app.command("status")
def server_status() -> None:
    """Show the running server's PID, if any."""
    status()


# ---------------------------------------------------------------------------
# sample workbook
# ---------------------------------------------------------------------------


@app.command("sample-workbook")
def sample_workbook(
    out: Path = typer.Option(
        Path("merton-sample.xlsx"),
        "--out",
        "-o",
        help="Destination .xlsx path.",
    ),
) -> None:
    """Write a worked-example workbook to ``out``."""
    from ...excel.sample import write_sample_workbook

    written = write_sample_workbook(out)
    console.print(f"[green]Sample workbook written[/] to {written}")
