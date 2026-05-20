"""NumPy / SciPy reference kernels.

These functions are the canonical source of truth for the maths. Other
backends (Numba, CuPy, JAX, MLX) must produce bit-identical or close-to-it
outputs on the same inputs.
"""

from __future__ import annotations

import numpy as np

SQRT_2PI = np.sqrt(2.0 * np.pi)


def norm_cdf(x: np.ndarray | float) -> np.ndarray:
    """Standard normal CDF Φ(x). Uses scipy.special.ndtr for accuracy + speed."""
    from scipy.special import ndtr  # lazy: scipy.special is cheap but tracks with scipy.stats

    return ndtr(x)


def norm_pdf(x: np.ndarray | float) -> np.ndarray:
    """Standard normal PDF φ(x)."""
    return np.exp(-0.5 * np.asarray(x, dtype=np.float64) ** 2) / SQRT_2PI


def norm_ppf(p: np.ndarray | float) -> np.ndarray:
    """Inverse standard normal CDF Φ⁻¹(p). Required for physical-PD mapping."""
    from scipy.stats import norm  # lazy: scipy.stats costs ~300 ms to import

    return norm.ppf(p)


def d1_d2(
    asset_value: np.ndarray | float,
    asset_vol: np.ndarray | float,
    debt: np.ndarray | float,
    rf: np.ndarray | float,
    T: np.ndarray | float,
    dividend_yield: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Black-Scholes-Merton ``d1`` and ``d2``.

    ::

        d1 = [ln(A/D) + (r - q + σ²/2) * T] / (σ * √T)
        d2 = d1 - σ * √T
    """
    A = np.asarray(asset_value, dtype=np.float64)
    s = np.asarray(asset_vol, dtype=np.float64)
    D = np.asarray(debt, dtype=np.float64)
    r = np.asarray(rf, dtype=np.float64)
    q = np.asarray(dividend_yield, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    sqrtT = np.sqrt(T_)
    d1 = (np.log(A / D) + (r - q + 0.5 * s * s) * T_) / (s * sqrtT)
    d2 = d1 - s * sqrtT
    return d1, d2


def equity_value(
    asset_value: np.ndarray | float,
    asset_vol: np.ndarray | float,
    debt: np.ndarray | float,
    rf: np.ndarray | float,
    T: np.ndarray | float,
    dividend_yield: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Black-Scholes-Merton equity-as-call-option value.

    ::

        E = A * exp(-q * T) * Φ(d1) - D * exp(-r * T) * Φ(d2)
    """
    d1, d2 = d1_d2(asset_value, asset_vol, debt, rf, T, dividend_yield)
    A = np.asarray(asset_value, dtype=np.float64)
    D = np.asarray(debt, dtype=np.float64)
    r = np.asarray(rf, dtype=np.float64)
    q = np.asarray(dividend_yield, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    return A * np.exp(-q * T_) * norm_cdf(d1) - D * np.exp(-r * T_) * norm_cdf(d2)


def distance_to_default_kernel(
    asset_value: np.ndarray | float,
    asset_vol: np.ndarray | float,
    debt: np.ndarray | float,
    rf: np.ndarray | float,
    T: np.ndarray | float,
    dividend_yield: np.ndarray | float = 0.0,
) -> np.ndarray:
    """``d2`` of Black-Scholes-Merton = distance to default."""
    _, d2 = d1_d2(asset_value, asset_vol, debt, rf, T, dividend_yield)
    return d2


def prob_of_default_kernel(dd: np.ndarray | float) -> np.ndarray:
    """Risk-neutral PD = Φ(-DD)."""
    return norm_cdf(-np.asarray(dd, dtype=np.float64))


__all__ = [
    "d1_d2",
    "distance_to_default_kernel",
    "equity_value",
    "norm_cdf",
    "norm_pdf",
    "norm_ppf",
    "prob_of_default_kernel",
]
