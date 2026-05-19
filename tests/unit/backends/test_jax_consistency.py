"""Cross-backend consistency for the JAX path.

Only runs when ``merton[jax]`` is installed.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# Import after the importorskip so we never load the JAX backend on hosts
# without JAX (CI matrix cells run JAX only where the extra is installed).
from merton._backend import _jax, _numpy  # noqa: E402


@pytest.mark.jax
class TestJAXKernelConsistency:
    def test_d1_d2_matches_numpy(self) -> None:
        A = np.array([80.0, 100.0, 150.0])
        s = np.array([0.20, 0.30, 0.25])
        D = np.array([50.0, 60.0, 80.0])
        r, T, q = 0.04, 1.0, 0.0
        d1_np, d2_np = _numpy.d1_d2(A, s, D, r, T, q)
        d1_jx, d2_jx = _jax.d1_d2(A, s, D, r, T, q)
        np.testing.assert_allclose(np.asarray(d1_jx), d1_np, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(d2_jx), d2_np, rtol=1e-6)

    def test_equity_value_matches_numpy(self) -> None:
        A = np.array([100.0, 200.0])
        s = np.array([0.25, 0.30])
        D = np.array([60.0, 100.0])
        r, T = 0.04, 1.0
        E_np = _numpy.equity_value(A, s, D, r, T, 0.0)
        E_jx = _jax.equity_value(A, s, D, r, T, 0.0)
        np.testing.assert_allclose(np.asarray(E_jx), E_np, rtol=1e-6)

    def test_dispatch_picks_jax_when_input_is_jax_array(self) -> None:
        from merton._backend import resolve

        x = jnp.array([1.0, 2.0, 3.0])
        assert resolve(x) == "jax"
