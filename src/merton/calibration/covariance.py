"""MLE asymptotic standard errors and delta-method propagation.

When a calibrator returns the observed Fisher information (the Hessian of the
negative log-likelihood at the MLE), this module turns it into:

- the asymptotic covariance of the parameter estimates,
- standard errors and Wald confidence intervals for each parameter,
- propagated standard errors / CIs for derived quantities (DD, PD, spread)
  via the delta method.

The delta method gives ``Var[g(θ)] ≈ J · Σ · J^T`` where ``J = ∂g/∂θ`` is the
Jacobian of ``g`` at ``θ̂`` and ``Σ`` is the asymptotic covariance of ``θ̂``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .._typing import FloatArray
from ..exceptions import MertonError

_EPS = 1e-8


@dataclass(slots=True, frozen=True)
class ConfInt:
    """Symmetric (or empirical) confidence interval for a single quantity."""

    lower: float
    upper: float
    level: float
    method: str
    se: float | None = None

    def __iter__(self):
        yield self.lower
        yield self.upper

    def as_tuple(self) -> tuple[float, float]:
        return (self.lower, self.upper)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"ConfInt(lower={self.lower:.6g}, upper={self.upper:.6g}, "
            f"level={self.level}, method={self.method!r})"
        )


def cov_from_hessian(hessian: np.ndarray) -> np.ndarray:
    """Invert an observed-Fisher-information matrix to a covariance.

    Adds a tiny ridge for numerical stability if the matrix is nearly
    singular; raises :class:`MertonError` if it can't be inverted at all.
    """
    H = np.asarray(hessian, dtype=np.float64)
    if H.shape[0] != H.shape[1]:
        raise MertonError("Hessian must be square")
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        ridge = _EPS * float(np.trace(H)) / H.shape[0]
        try:
            cov = np.linalg.inv(H + ridge * np.eye(H.shape[0]))
        except np.linalg.LinAlgError as err:  # pragma: no cover - degenerate
            raise MertonError(
                "could not invert Hessian to obtain MLE covariance",
                suggested_fix="check that the optimisation converged",
            ) from err
    return cov


def standard_errors(cov: np.ndarray) -> FloatArray:
    """Per-parameter standard errors = sqrt(diag(Σ))."""
    diag = np.diag(np.asarray(cov, dtype=np.float64))
    return np.sqrt(np.clip(diag, 0.0, np.inf))


def wald_ci(
    estimate: float,
    se: float,
    *,
    level: float = 0.95,
    method: str = "asymptotic",
) -> ConfInt:
    """Symmetric Wald CI ``estimate ± z · se``."""
    if not 0 < level < 1:
        raise MertonError("level must be in (0, 1)")
    from scipy.stats import norm  # lazy: scipy.stats costs ~300 ms to import

    z = float(norm.ppf(0.5 + level / 2.0))
    return ConfInt(
        lower=float(estimate) - z * float(se),
        upper=float(estimate) + z * float(se),
        level=level,
        method=method,
        se=float(se),
    )


def delta_method(
    g: Callable[[np.ndarray], float],
    theta: np.ndarray,
    cov: np.ndarray,
    *,
    eps: float | None = None,
) -> tuple[float, float]:
    """Apply the delta method to a scalar function ``g`` of the parameters.

    Returns ``(g(theta), se_g)`` where ``se_g² = J · Σ · Jᵀ`` and ``J`` is the
    gradient of ``g`` computed by central finite differences.
    """
    theta = np.asarray(theta, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    if eps is None:
        eps = float(np.maximum(1e-6, 1e-4 * np.linalg.norm(theta) + 1e-12))
    grad = np.empty_like(theta)
    base = float(g(theta))
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = eps
        grad[i] = (float(g(theta + step)) - float(g(theta - step))) / (2.0 * eps)
    var = float(grad @ cov @ grad)
    return base, float(np.sqrt(max(var, 0.0)))


__all__ = [
    "ConfInt",
    "cov_from_hessian",
    "delta_method",
    "standard_errors",
    "wald_ci",
]
