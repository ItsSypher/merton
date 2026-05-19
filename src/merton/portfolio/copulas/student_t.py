"""Student-``t`` copula sampler.

Heavier-tailed than the Gaussian copula — induces tail dependence between
defaults, which is what regulators worry about most in stress scenarios.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as _student_t

from ..._typing import FloatArray
from ...exceptions import MertonInputError


class TCopula:
    """Multivariate Student-``t`` copula with degrees-of-freedom ``df``."""

    def __init__(
        self,
        correlation: float | FloatArray,
        n_firms: int | None = None,
        *,
        df: float = 4.0,
    ) -> None:
        if df <= 2:
            raise MertonInputError("df must exceed 2 for finite variance")
        self.df = float(df)
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
        try:
            self._chol = np.linalg.cholesky(self.corr)
        except np.linalg.LinAlgError as err:  # pragma: no cover
            raise MertonInputError("correlation matrix is not positive-definite") from err
        self.n = self.corr.shape[0]

    def sample(self, n: int, *, rng: np.random.Generator | None = None) -> FloatArray:
        """Draw ``n`` samples; returns shape ``(n, n_firms)``."""
        if rng is None:
            rng = np.random.default_rng()
        z = rng.standard_normal(size=(n, self.n))
        chi2 = rng.chisquare(self.df, size=n) / self.df
        # t = Z / sqrt(chi2/df), each row scaled by 1/sqrt(chi2/df)
        x = (z @ self._chol.T) / np.sqrt(chi2)[:, None]
        return _student_t.cdf(x, df=self.df)


__all__ = ["TCopula"]
