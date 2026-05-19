"""``merton config`` — inspect and modify the persisted configuration."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import tomli_w
import typer
from platformdirs import user_config_dir
from rich.console import Console
from rich.table import Table

from ..._config import MertonSettings, get_config, reset_config

app = typer.Typer(no_args_is_help=True)
console = Console()


def _config_path() -> Path:
    """Return the on-disk path of the user config.

    Honours the ``MERTON_CONFIG_DIR`` env var so tests (and users with a
    preferred location) can redirect the file without depending on the
    platform-specific default.
    """
    override = os.environ.get("MERTON_CONFIG_DIR")
    if override:
        return Path(override) / "config.toml"
    return Path(user_config_dir("merton")) / "config.toml"


def _read_toml() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text())


def _write_toml(data: dict[str, Any]) -> Path:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tomli_w.dumps(data).encode())
    return path


@app.command("show")
def show() -> None:
    """Print the resolved configuration (defaults → file → env)."""
    cfg = get_config()
    tbl = Table(title="merton settings", title_justify="left")
    tbl.add_column("Field", style="bold cyan")
    tbl.add_column("Value")
    for field_name in MertonSettings.model_fields:
        tbl.add_row(field_name, repr(getattr(cfg, field_name)))
    console.print(tbl)
    console.print(f"\nConfig file: [dim]{_config_path()}[/]")


@app.command("set")
def set_(
    key: str = typer.Argument(..., help="Setting name (e.g. backend, default_horizon)."),
    value: str = typer.Argument(..., help="New value (parsed as TOML literal)."),
) -> None:
    """Persist ``key = value`` into the user config file."""
    if key not in MertonSettings.model_fields:
        console.print(
            f"[red]Unknown key {key!r}.[/] Use `merton config show` to see available fields."
        )
        raise typer.Exit(code=1)
    # Parse the value as a TOML literal so types are preserved.
    parsed = tomllib.loads(f"_v = {value}")["_v"]
    data = _read_toml()
    data[key] = parsed
    path = _write_toml(data)
    reset_config()
    console.print(f"Set [bold]{key}[/] = {parsed!r} ([dim]{path}[/])")


@app.command("reset")
def reset() -> None:
    """Remove the user config file (revert to defaults + env vars)."""
    path = _config_path()
    if path.exists():
        path.unlink()
        console.print(f"Removed [bold]{path}[/]")
    else:
        console.print("[dim]No config file to remove.[/]")
    reset_config()
