"""When CuPy is requested but unavailable, dispatch must emit the warning."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from merton._backend import _registry, resolve
from merton.exceptions import MertonBackendFallbackWarning


@pytest.fixture(autouse=True)
def _reset_registry():
    _registry.reset()
    yield
    _registry.reset()


def test_cupy_unavailable_falls_back() -> None:
    if importlib.util.find_spec("cupy") is not None:  # pragma: no cover
        pytest.skip("cupy present; this test only runs without the extra")
    with pytest.warns(MertonBackendFallbackWarning):
        chosen = resolve(np.array([1.0]), backend="cupy")
    assert chosen in {"numba", "numpy"}


@pytest.mark.gpu
def test_cupy_dispatch_chosen_when_available() -> None:
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("cupy not installed")
    import cupy as cp  # type: ignore[import-not-found]

    x = cp.array([1.0, 2.0, 3.0])
    # Auto-detection by namespace.
    assert resolve(x) == "cupy"


@pytest.mark.gpu
def test_cupy_kernels_match_numpy() -> None:
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("cupy not installed")
    import cupy as cp  # type: ignore[import-not-found]

    from merton._backend import _cupy, _numpy

    A = np.array([100.0, 150.0, 200.0])
    s = np.array([0.25, 0.30, 0.35])
    D = np.array([60.0, 80.0, 100.0])
    r, T, q = 0.04, 1.0, 0.0
    d1_np, d2_np = _numpy.d1_d2(A, s, D, r, T, q)
    d1_cp, d2_cp = _cupy.d1_d2(cp.asarray(A), cp.asarray(s), cp.asarray(D), r, T, q)
    np.testing.assert_allclose(cp.asnumpy(d1_cp), d1_np, rtol=1e-7)
    np.testing.assert_allclose(cp.asnumpy(d2_cp), d2_np, rtol=1e-7)
