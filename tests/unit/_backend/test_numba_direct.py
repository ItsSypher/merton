"""Direct calls to the Numba-jitted entry points.

These ensure the inner ``_phi_scalar`` / loop body / ``_d1_d2_impl`` paths
get exercised by the coverage runner (Numba's ``@njit`` wrappers and inner
loops aren't traced through the standard collector unless they're called
directly).
"""

from __future__ import annotations

import numpy as np

from merton._backend import _numba


class TestDirectNumbaPaths:
    def test_norm_cdf_array_path(self) -> None:
        x = np.linspace(-3.0, 3.0, 16)
        out = _numba.norm_cdf(x)
        # Symmetry around zero.
        np.testing.assert_allclose(out + out[::-1], np.ones_like(out), rtol=1e-6)

    def test_norm_pdf_peak_at_zero(self) -> None:
        x = np.linspace(-2.0, 2.0, 9)
        pdf = _numba.norm_pdf(x)
        # The middle index (x ≈ 0) is the maximum.
        assert np.argmax(pdf) == len(pdf) // 2

    def test_d1_d2_array_inputs(self) -> None:
        A = np.linspace(80.0, 200.0, 12)
        s = np.full(12, 0.25)
        D = np.full(12, 60.0)
        r = np.full(12, 0.04)
        T = np.full(12, 1.0)
        q = np.zeros(12)
        d1, d2 = _numba._d1_d2_impl(A, s, D, r, q, T)
        assert d1.shape == (12,)
        np.testing.assert_allclose(d1 - d2, s * np.sqrt(T), rtol=1e-7)
