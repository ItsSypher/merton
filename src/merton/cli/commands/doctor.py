"""``merton doctor`` — print a runtime diagnostic.

Reports:

- Python version and free-threaded (no-GIL) status.
- Installed array backends and which one ``auto`` would resolve to.
- GPU / CUDA / Apple-Silicon / JAX device availability.
- Versions of the major dependencies.
- Hints for missing optional extras.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from importlib.util import find_spec

import typer
from rich.console import Console
from rich.table import Table

from ... import __version__
from ..._backend import _registry

console = Console()


def _is_free_threaded() -> bool:
    """Return True on a free-threaded (PEP 703) build of Python 3.13+/3.14+."""
    # 3.13+ exposes sys._is_gil_enabled(); older interpreters always have the GIL.
    fn = getattr(sys, "_is_gil_enabled", None)
    if fn is None:
        return False
    try:
        return not fn()
    except Exception:  # pragma: no cover
        return False


def _safe_version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "—"


def _cuda_devices() -> list[str]:
    """Best-effort enumeration of installed NVIDIA GPUs via CuPy."""
    if find_spec("cupy") is None:
        return []
    try:
        cp = importlib.import_module("cupy")
        n = cp.cuda.runtime.getDeviceCount()
        return [cp.cuda.runtime.getDeviceProperties(i)["name"].decode() for i in range(n)]
    except Exception:  # pragma: no cover
        return []


def _jax_devices() -> list[str]:
    if find_spec("jax") is None:
        return []
    try:
        jax = importlib.import_module("jax")
        return [str(d) for d in jax.devices()]
    except Exception:  # pragma: no cover
        return []


def _mlx_available() -> bool:
    return find_spec("mlx") is not None and platform.system() == "Darwin"


def run() -> None:
    """Print runtime diagnostics."""
    console.rule(f"[bold]merton {__version__}")

    # Python / build
    info = Table(show_header=False, box=None)
    info.add_column("k", style="bold cyan")
    info.add_column("v")
    info.add_row("Python", f"{platform.python_version()} ({platform.python_implementation()})")
    info.add_row("Free-threaded (no-GIL)", "yes" if _is_free_threaded() else "no")
    info.add_row("Platform", f"{platform.system()} {platform.release()} ({platform.machine()})")
    console.print(info)

    # Backends
    backends = Table(title="Array backends", title_justify="left")
    backends.add_column("Backend")
    backends.add_column("Installed")
    backends.add_column("Notes")
    avail = set(_registry.available())
    default = _registry.default_backend()
    for name in ("numpy", "numba", "cupy", "jax", "mlx"):
        is_installed = name in avail
        note_parts = []
        if name == default:
            note_parts.append("[bold green]default[/]")
        if name == "cupy":
            devs = _cuda_devices()
            if devs:
                note_parts.append(f"GPUs: {', '.join(devs)}")
        if name == "jax":
            jdevs = _jax_devices()
            if jdevs:
                note_parts.append(f"devices: {', '.join(jdevs)}")
        if name == "mlx" and not _mlx_available() and is_installed:
            note_parts.append("only useful on Apple Silicon")
        backends.add_row(
            name,
            "[green]✓[/]" if is_installed else "[dim]—[/]",
            " · ".join(note_parts) or "",
        )
    console.print(backends)

    # Key dependencies
    deps = Table(title="Key dependencies", title_justify="left")
    deps.add_column("Package")
    deps.add_column("Version")
    for pkg in (
        "numpy",
        "scipy",
        "pandas",
        "pyarrow",
        "numba",
        "joblib",
        "structlog",
        "pydantic",
        "pydantic-settings",
        "typer",
        "rich",
        "jax",
        "jaxlib",
        "cupy-cuda12x",
        "mlx",
        "xlwings",
    ):
        deps.add_row(pkg, _safe_version(pkg))
    console.print(deps)

    # Hints
    hints: list[str] = []
    if "jax" not in avail:
        hints.append(r"Install [bold]merton\[jax][/] for autodiff Greeks and GPU/TPU acceleration.")
    if "cupy" not in avail and platform.system() != "Darwin":
        hints.append(r"Install [bold]merton\[gpu][/] (CUDA 12) for GPU-accelerated panels.")
    if platform.system() == "Darwin" and platform.machine() == "arm64" and "mlx" not in avail:
        hints.append(
            r"Install [bold]merton\[mlx][/] for Metal-accelerated kernels on Apple Silicon."
        )
    if not _is_free_threaded() and sys.version_info >= (3, 13):
        hints.append("Try a [bold]free-threaded[/] Python (cp313t / cp314t) for parallel panels.")
    if hints:
        console.print("\n[bold]Suggestions[/]")
        for h in hints:
            console.print(f"  • {h}")


def _entrypoint(ctx: typer.Context) -> None:  # pragma: no cover - typer adapter
    run()
