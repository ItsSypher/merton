"""Cold-import budget regression test.

The 1.0 release ships with `import merton` at ~500 ms on a 2024
M-series workstation. If a future change re-introduces an eager heavy
import (scipy.stats, pandas, jax, etc.), this test will fire so we
catch the regression in CI rather than in user bug reports.

Budget rationale
----------------
- numpy + numba together cost ~150 ms (hard deps; can't be deferred).
- pydantic + pydantic-settings cost ~60 ms.
- structlog + typer + rich + joblib together cost ~80 ms.
- Dev workstations hit ~500 ms; GH-Actions cold runners are noticeably
  slower (saw 1670 ms on ubuntu-latest py3.11). Budget = 2500 ms so we
  still catch the original 1100 ms class of regression (eager
  backtest/pandas) without false positives on the cold runners.
"""

from __future__ import annotations

import subprocess
import sys

# Budget in milliseconds. Tight enough to catch regressions; loose enough
# to absorb GH-Actions variance (Windows / fresh ubuntu cold-start can
# be 3-4× slower than warm macOS dev machines).
COLD_IMPORT_BUDGET_MS = 2500.0


def _measure_cold_import_ms() -> float:
    """Spawn a fresh interpreter and time `import merton`.

    A subprocess is required: the parent process has already imported
    merton (via collection), so an in-process timer would always return
    the cached re-import cost (~microseconds), not the actual cold cost.
    """
    code = (
        "import time;"
        "t0=time.perf_counter();"
        "import merton;"
        "print(f'{(time.perf_counter()-t0)*1000:.3f}')"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    return float(out)


def test_cold_import_under_budget() -> None:
    # Run twice and take the minimum: the first invocation can stall on
    # filesystem caches (especially on macOS APFS) — the second measures
    # warm-OS, cold-Python.
    samples = [_measure_cold_import_ms() for _ in range(3)]
    best = min(samples)
    assert best < COLD_IMPORT_BUDGET_MS, (
        f"Cold import of `merton` took {best:.1f} ms — over the {COLD_IMPORT_BUDGET_MS:.0f} ms "
        f"budget. All samples: {samples}. Likely cause: a new top-level "
        f"`import` of a heavy dependency (scipy.stats, pandas, jax, …) "
        f"that should be deferred to the function that uses it."
    )


def test_top_level_does_not_pull_pandas() -> None:
    """`import merton` must not load pandas (it's lazy via batch_fit/result)."""
    code = "import sys, merton;print('pandas' if 'pandas' in sys.modules else 'no-pandas')"
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "no-pandas", (
        "Cold `import merton` pulled pandas into sys.modules. "
        "Pandas is a heavy import (~160 ms) and should stay lazy."
    )


def test_top_level_does_not_pull_jax_or_cupy() -> None:
    """`import merton` must not load any optional backend (jax, cupy, mlx)."""
    code = (
        "import sys, merton;"
        "leaks = [m for m in ('jax', 'cupy', 'mlx') if m in sys.modules];"
        "print(','.join(leaks) or 'clean')"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "clean", (
        f"Cold `import merton` pulled optional backends into sys.modules: {out}. "
        f"Optional extras must stay opt-in."
    )


# Note: scipy.stats is currently pulled in by `merton.calibration.kmv_iterative`
# and is the largest remaining cold-import contributor (~200 ms). Restructuring
# the calibrator registry to lazy-populate is a 1.x roadmap item; for now the
# overall budget test above guards against regression in the other direction.
