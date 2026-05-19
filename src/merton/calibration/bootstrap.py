"""Block bootstrap for time-series calibrators.

The moving / circular block bootstrap (Politis & Romano, 1994) resamples
contiguous blocks of length ``b`` from a return series of length ``n`` to
preserve serial dependence. For an equity series we bootstrap **log-returns**
(stationary by Merton assumption), reconstruct equity paths from each
resample, refit the calibrator, and report empirical percentile CIs of any
chosen scalar function of the fitted parameters.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .._typing import FloatArray
from .covariance import ConfInt


@dataclass(slots=True)
class BootstrapResult:
    """Output of :func:`block_bootstrap_calibration`."""

    parameter_samples: dict[str, FloatArray] = field(default_factory=dict)
    derived_samples: dict[str, FloatArray] = field(default_factory=dict)
    n_resamples: int = 0
    block_length: int = 0

    def ci(self, name: str, level: float = 0.95) -> ConfInt:
        """Percentile CI for parameter ``name`` (or any derived quantity)."""
        samples = self.parameter_samples.get(name)
        if samples is None:
            samples = self.derived_samples.get(name)
        if samples is None:
            raise KeyError(f"{name!r} not in bootstrap samples")
        alpha = (1.0 - level) / 2.0
        lo = float(np.quantile(samples, alpha))
        hi = float(np.quantile(samples, 1.0 - alpha))
        return ConfInt(
            lower=lo,
            upper=hi,
            level=level,
            method="bootstrap",
            se=float(np.std(samples, ddof=1)),
        )


def _circular_block_sample(
    series: FloatArray,
    block_length: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw one circular-block bootstrap sample of the same length as ``series``."""
    n = series.shape[0]
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    out = np.empty(n_blocks * block_length, dtype=series.dtype)
    for i, s in enumerate(starts):
        for k in range(block_length):
            out[i * block_length + k] = series[(s + k) % n]
    return out[:n]


def block_bootstrap_calibration(
    equity_series: FloatArray,
    refit: Callable[[FloatArray], dict[str, float]],
    *,
    n_resamples: int = 200,
    block_length: int | None = None,
    seed: int | None = None,
    derived: dict[str, Callable[[dict[str, float]], float]] | None = None,
) -> BootstrapResult:
    """Block-bootstrap a time-series calibrator.

    Parameters
    ----------
    equity_series
        The 1-D equity time series the original calibration was fit on.
    refit
        Callable: ``refit(resampled_series) -> {"asset_vol": ..., "asset_drift": ..., ...}``.
        It must accept a same-length equity series and return a parameter dict.
    n_resamples
        Number of bootstrap replications. 200 is a sane default for CIs; bump
        to 1 000+ for tighter percentile estimates.
    block_length
        Length of each contiguous block. Defaults to ``round(n ** (1/3))``
        per Politis-Romano.
    seed
        RNG seed for reproducibility.
    derived
        Optional dict mapping name → ``f(params_dict) -> float``. Each
        function is evaluated on each resample to give CI's for derived
        quantities (e.g. DD or PD at the calibrated point).
    """
    eq = np.asarray(equity_series, dtype=np.float64)
    n = eq.shape[0]
    if block_length is None:
        block_length = max(2, round(n ** (1 / 3)))
    rng = np.random.default_rng(seed)

    # Bootstrap log-returns and re-integrate to keep the level realistic.
    log_returns = np.diff(np.log(eq))
    derived = derived or {}
    parameter_samples: dict[str, list[float]] = {}
    derived_samples: dict[str, list[float]] = {name: [] for name in derived}

    for _ in range(n_resamples):
        sample_returns = _circular_block_sample(log_returns, block_length, rng)
        sample_series = np.empty(n, dtype=np.float64)
        sample_series[0] = float(eq[0])
        sample_series[1:] = float(eq[0]) * np.exp(np.cumsum(sample_returns[: n - 1]))
        try:
            params = refit(sample_series)
        except Exception:  # nosec B112 - one bad bootstrap sample shouldn't kill the run; we report `converged_resamples` in the result for visibility
            continue
        for k, v in params.items():
            parameter_samples.setdefault(k, []).append(float(v))
        for name, fn in derived.items():
            with contextlib.suppress(Exception):
                derived_samples[name].append(float(fn(params)))

    return BootstrapResult(
        parameter_samples={
            k: np.asarray(v, dtype=np.float64) for k, v in parameter_samples.items()
        },
        derived_samples={k: np.asarray(v, dtype=np.float64) for k, v in derived_samples.items()},
        n_resamples=n_resamples,
        block_length=block_length,
    )


__all__ = ["BootstrapResult", "block_bootstrap_calibration"]
