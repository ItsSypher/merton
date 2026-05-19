"""Top-level :mod:`typer` application wiring up the merton subcommands."""

from __future__ import annotations

import typer

from .. import __version__
from .commands import config as config_cmd
from .commands import doctor as doctor_cmd
from .commands import excel as excel_cmd
from .commands import fit as fit_cmd

app = typer.Typer(
    name="merton",
    no_args_is_help=True,
    add_completion=True,
    help="Merton structural credit-risk model CLI.",
    pretty_exceptions_show_locals=False,
)

app.add_typer(config_cmd.app, name="config", help="Read or write package configuration.")
app.add_typer(excel_cmd.app, name="excel", help="Excel integration (install, server, sample).")
app.command(name="doctor", help="Diagnose backends, GPU availability, and runtime info.")(
    doctor_cmd.run
)
app.command(name="fit", help="Fit a single firm or panel from a CSV/Parquet input.")(fit_cmd.run)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"merton {__version__}")
        raise typer.Exit


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Print version and exit.",
    ),
) -> None:
    """Merton structural credit-risk model."""


if __name__ == "__main__":  # pragma: no cover
    app()
