"""Gaussian copula sampler."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..._typing import FloatArray
from ...exceptions import MertonInputError


class GaussianCopula:
    r"""Multivariate Gaussian copula.

    Samples ``u_i ∈ [0, 1]`` from a multivariate Gaussian with the supplied
    correlation matrix, then maps to uniform marginals via the normal CDF.
    These uniforms are then used to drive obligor default indicators.

    Parameters
    ----------
    correlation
        Either a scalar (constant pairwise correlation) or a
        ``(n_firms, n_firms)`` matrix.
    n_firms
        When ``correlation`` is a scalar, this sets the dimension.
    """

    def __init__(self, correlation: float | FloatArray, n_firms: int | None = None) -> None:
        if np.isscalar(correlation):
            if n_firms is None:
                raise MertonInputError("n_firms required when correlation is scalar")
            rho = float(correlation)
            if not -1.0 < rho < 1.0:
                raise MertonInputError("scalar correlation must lie in (-1, 1)")
            cov = np.full((n_firms, n_firms), rho)
            np.fill_diagonal(cov, 1.0)
            self.corr = cov
        else:
            corr = np.asarray(correlation, dtype=np.float64)
            if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
                raise MertonInputError("correlation matrix must be square 2-D")
            self.corr = corr
        # Cholesky for fast sampling.
        try:
            self._chol = np.linalg.cholesky(self.corr)
        except np.linalg.LinAlgError as err:  # pragma: no cover - degenerate
            raise MertonInputError(
                "correlation matrix is not positive-definite",
                suggested_fix="Apply nearest-correlation regularisation before passing in.",
            ) from err
        self.n = self.corr.shape[0]

    def sample(self, n: int, *, rng: np.random.Generator | None = None) -> FloatArray:
        """Draw ``n`` independent samples; returns shape ``(n, n_firms)``."""
        if rng is None:
            rng = np.random.default_rng()
        z = rng.standard_normal(size=(n, self.n))
        x = z @ self._chol.T
        # Marginals: each column is N(0, 1); CDF gives uniforms.
        return norm.cdf(x)

    def sample_normal(self, n: int, *, rng: np.random.Generator | None = None) -> FloatArray:
        """Draw the *latent normals* directly (skip the CDF transform)."""
        if rng is None:
            rng = np.random.default_rng()
        z = rng.standard_normal(size=(n, self.n))
        return z @ self._chol.T


__all__ = ["GaussianCopula"]
