"""Entry point so ``python -m merton`` invokes the typer CLI."""

from __future__ import annotations

from .cli.main import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
