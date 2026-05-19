"""Numerical solvers shared by calibration backends."""

from __future__ import annotations

import numpy as np

from .._backend._numpy import d1_d2, norm_cdf
from ..exceptions import CalibrationConvergenceError


def two_eq_residuals(
    A: float,
    sigma_A: float,
    *,
    E: float,
    sigma_E: float,
    D: float,
    r: float,
    T: float,
    q: float = 0.0,
) -> tuple[float, float]:
    """The two-equation residuals (E-equation, σ_E-equation) at (A, σ_A).

    ``f₁ = A·e^(-qT)·Φ(d₁) - D·e^(-rT)·Φ(d₂) - E``
    ``f₂ = e^(-qT)·Φ(d₁)·σ_A·A/E - σ_E``
    """
    d1, d2 = d1_d2(A, sigma_A, D, r, T, q)
    f1 = A * np.exp(-q * T) * norm_cdf(d1) - D * np.exp(-r * T) * norm_cdf(d2) - E
    f2 = np.exp(-q * T) * norm_cdf(d1) * sigma_A * A / E - sigma_E
    return float(f1), float(f2)


def solve_two_equation(
    *,
    E: float,
    sigma_E: float,
    D: float,
    r: float,
    T: float,
    q: float = 0.0,
    A0: float | None = None,
    sigma_A0: float | None = None,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> tuple[float, float, int, bool]:
    """Solve the standard two-equation JMR system for (A, σ_A).

    Uses :func:`scipy.optimize.fsolve` (Powell hybrid). Initial guesses come
    from the naive proxies ``A ≈ E + D``, ``σ_A ≈ σ_E · E / (E + D)``.

    Returns
    -------
    asset_value, asset_vol, n_iter, converged
    """
    from scipy.optimize import fsolve

    if A0 is None:
        A0 = E + D
    if sigma_A0 is None:
        sigma_A0 = sigma_E * E / max(A0, 1e-12)

    iters = {"n": 0}

    def residuals(x: np.ndarray) -> np.ndarray:
        iters["n"] += 1
        a, s = float(x[0]), float(x[1])
        if a <= 0 or s <= 0:
            return np.array([1e6, 1e6])
        f1, f2 = two_eq_residuals(a, s, E=E, sigma_E=sigma_E, D=D, r=r, T=T, q=q)
        return np.array([f1, f2])

    x0 = np.array([A0, sigma_A0])
    sol, _info, ier, msg = fsolve(residuals, x0, xtol=tol, maxfev=max_iter, full_output=True)
    converged = ier == 1
    if not converged:
        # As a last resort, try a fresh bracketed search around naive guesses.
        for scale_a, scale_s in [(0.5, 0.5), (1.5, 1.5), (2.0, 0.7)]:
            x0_alt = np.array([A0 * scale_a, sigma_A0 * scale_s])
            sol_alt, _, ier_alt, _ = fsolve(
                residuals, x0_alt, xtol=tol, maxfev=max_iter, full_output=True
            )
            if ier_alt == 1:
                sol, ier, converged = sol_alt, ier_alt, True
                break
    if not converged:
        raise CalibrationConvergenceError(
            f"two-equation solver failed: {msg}",
            suggested_fix="Increase max_iter, relax tol, or pre-process equity_vol.",
        )
    return float(sol[0]), float(sol[1]), iters["n"], converged


__all__ = ["solve_two_equation", "two_eq_residuals"]
