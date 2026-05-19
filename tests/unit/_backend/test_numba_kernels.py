"""Direct tests of the Numba kernels.

The numpy-vs-numba consistency test in tests/unit/core/test_distance.py
exercises ``distance_to_default_kernel``; this file adds direct coverage
for ``norm_pdf``, ``equity_value``, ``prob_of_default_kernel``, scalar
broadcasting, and ``warm_cache``.
"""

from __future__ import annotations

import numpy as np

from merton._backend import _numba, _numpy


class TestNumbaKernels:
    def test_norm_pdf_matches_numpy(self) -> None:
        x = np.linspace(-3.0, 3.0, 32)
        out_nb = _numba.norm_pdf(x)
        out_np = _numpy.norm_pdf(x)
        np.testing.assert_allclose(out_nb, out_np, rtol=1e-6)

    def test_equity_value_matches_numpy(self) -> None:
        A = np.array([80.0, 100.0, 150.0])
        s = np.array([0.20, 0.25, 0.35])
        D = np.array([50.0, 60.0, 80.0])
        r = 0.04
        T = 1.0
        np.testing.assert_allclose(
            _numba.equity_value(A, s, D, r, T, 0.0),
            _numpy.equity_value(A, s, D, r, T, 0.0),
            rtol=1e-6,
        )

    def test_prob_of_default_scalar(self) -> None:
        # Scalar input should give a scalar-like output.
        out = _numba.prob_of_default_kernel(2.0)
        np.testing.assert_allclose(float(out), float(_numpy.prob_of_default_kernel(2.0)), rtol=1e-6)

    def test_prob_of_default_array(self) -> None:
        dds = np.array([-1.0, 0.0, 1.0, 3.0])
        np.testing.assert_allclose(
            _numba.prob_of_default_kernel(dds),
            _numpy.prob_of_default_kernel(dds),
            rtol=1e-6,
        )

    def test_d1_d2_scalar_returns_scalar(self) -> None:
        d1, d2 = _numba.d1_d2(100.0, 0.25, 60.0, 0.04, 1.0)
        # 0-d numpy items.
        assert np.ndim(d1) == 0
        assert np.ndim(d2) == 0

    def test_d1_d2_broadcast_against_numpy(self) -> None:
        A = np.array([100.0, 150.0])
        s = 0.25
        D = np.array([60.0, 80.0])
        d1_nb, d2_nb = _numba.d1_d2(A, s, D, 0.04, 1.0)
        d1_np, d2_np = _numpy.d1_d2(A, s, D, 0.04, 1.0)
        np.testing.assert_allclose(d1_nb, d1_np, rtol=1e-6)
        np.testing.assert_allclose(d2_nb, d2_np, rtol=1e-6)

    def test_warm_cache_runs(self) -> None:
        _numba.warm_cache()  # smoke-test; just must not raise.
