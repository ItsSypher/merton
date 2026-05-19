r"""Vasicek (1987, 2002) single-factor portfolio model.

Used by Basel II/III IRB regulations to set capital requirements without
running a Monte Carlo. Each obligor's default indicator is generated from
a latent variable

.. math::

    Y_i = \sqrt{\rho}\, M + \sqrt{1 - \rho}\, \varepsilon_i,

with ``M, ε_i ~ N(0, 1)`` independent. Obligor ``i`` defaults when
``Y_i < \Phi^{-1}(\mathrm{PD}_i)``.

In the homogeneous limit (a portfolio of infinitely many infinitesimally
small loans with identical ``PD``, ``LGD``, ``ρ``), the loss fraction has
the closed-form CDF

.. math::

    F_L(x) = \Phi\!\left(\frac{\sqrt{1-\rho}\, \Phi^{-1}(x) - \Phi^{-1}(\mathrm{PD})}{\sqrt{\rho}}\right),

and the ``α``-quantile (Basel IRB confidence level: ``α = 0.999``) is

.. math::

    L_\alpha = \Phi\!\left(\frac{\Phi^{-1}(\mathrm{PD}) + \sqrt{\rho}\, \Phi^{-1}(\alpha)}{\sqrt{1-\rho}}\right).

References
----------
Vasicek, O. (2002). Loan Portfolio Value. *Risk* 15 (12), 160-162.

Basel Committee on Banking Supervision (2005). *An Explanatory Note on the
Basel II IRB Risk Weight Functions.*
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .._typing import ArrayLike, FloatArray
from ..exceptions import MertonInputError


def _validate_unit(name: str, x: ArrayLike) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    if np.any(arr < 0) or np.any(arr > 1):
        raise MertonInputError(f"{name} must lie in [0, 1]")
    return arr


def vasicek_loss_cdf(x: ArrayLike, pd: ArrayLike, rho: ArrayLike) -> FloatArray:
    r"""``F_L(x) = Φ((√(1-ρ) Φ⁻¹(x) - Φ⁻¹(PD)) / √ρ)``."""
    x_arr = _validate_unit("x", x)
    pd_arr = _validate_unit("pd", pd)
    rho_arr = _validate_unit("rho", rho)
    # Clip for numerical safety.
    x_clip = np.clip(x_arr, 1e-15, 1.0 - 1e-15)
    pd_clip = np.clip(pd_arr, 1e-15, 1.0 - 1e-15)
    rho_clip = np.clip(rho_arr, 1e-15, 1.0 - 1e-15)
    return norm.cdf(
        (np.sqrt(1.0 - rho_clip) * norm.ppf(x_clip) - norm.ppf(pd_clip)) / np.sqrt(rho_clip)
    )


def vasicek_var(pd: ArrayLike, rho: ArrayLike, alpha: float = 0.999) -> FloatArray:
    r"""``α``-quantile of the asymptotic loss distribution."""
    pd_arr = _validate_unit("pd", pd)
    rho_arr = _validate_unit("rho", rho)
    if not 0.5 < alpha < 1:
        raise MertonInputError("alpha must lie in (0.5, 1)")
    pd_clip = np.clip(pd_arr, 1e-15, 1.0 - 1e-15)
    rho_clip = np.clip(rho_arr, 1e-15, 1.0 - 1e-15)
    return norm.cdf(
        (norm.ppf(pd_clip) + np.sqrt(rho_clip) * norm.ppf(alpha)) / np.sqrt(1.0 - rho_clip)
    )


def basel_irb_capital(
    pd: ArrayLike,
    lgd: ArrayLike = 0.45,
    rho: ArrayLike | None = None,
    maturity: float = 1.0,
    *,
    alpha: float = 0.999,
    apply_maturity_adjustment: bool = True,
    asset_class: str = "corporate",
) -> FloatArray:
    r"""Basel III IRB unexpected-loss capital requirement.

    For corporate exposures, the BCBS prescribes

    .. math::

        \rho(\mathrm{PD}) = 0.12\,\frac{1 - e^{-50\,\mathrm{PD}}}{1 - e^{-50}}
                         + 0.24\,\left(1 - \frac{1 - e^{-50\,\mathrm{PD}}}{1 - e^{-50}}\right),

    with a maturity adjustment

    .. math::

        b(\mathrm{PD}) = (0.11852 - 0.05478\,\ln(\mathrm{PD}))^2,\quad
        \mathrm{MA}(M) = \frac{1 + (M - 2.5)\,b(\mathrm{PD})}{1 - 1.5\,b(\mathrm{PD})}.

    The capital requirement per unit of exposure is

    .. math::

        K = \mathrm{LGD}\,\left(L_\alpha(\mathrm{PD}, \rho) - \mathrm{PD}\right)\,\mathrm{MA}(M).
    """
    pd_arr = _validate_unit("pd", pd)
    lgd_arr = _validate_unit("lgd", lgd)
    if rho is None:
        rho_arr = basel_irb_correlation(pd_arr, asset_class=asset_class)
    else:
        rho_arr = _validate_unit("rho", rho)
    var = vasicek_var(pd_arr, rho_arr, alpha=alpha)
    unexpected_loss = lgd_arr * (var - pd_arr)
    if apply_maturity_adjustment:
        b = (0.11852 - 0.05478 * np.log(np.clip(pd_arr, 1e-15, 1.0))) ** 2
        ma = (1.0 + (maturity - 2.5) * b) / (1.0 - 1.5 * b)
        return unexpected_loss * ma
    return unexpected_loss


# Re-exported by the package; defined here to avoid circular import.
def basel_irb_correlation(pd: ArrayLike, *, asset_class: str = "corporate") -> FloatArray:
    """Basel III asset correlation as a function of PD and asset class."""
    pd_arr = _validate_unit("pd", pd)
    if asset_class == "corporate":
        lo, hi = 0.12, 0.24
    elif asset_class == "sme":
        # SME size-adjusted; assumes a $50M+ annual sales firm for the cap.
        lo, hi = 0.12, 0.24
    elif asset_class == "retail":
        lo, hi = 0.03, 0.16
    elif asset_class == "qrre":
        lo, hi = 0.04, 0.04  # qualifying revolving retail = constant 4%.
    else:
        raise MertonInputError(
            f"unknown asset_class {asset_class!r}",
            suggested_fix="Use one of: corporate, sme, retail, qrre.",
        )
    w = (1.0 - np.exp(-50.0 * pd_arr)) / (1.0 - np.exp(-50.0))
    return lo * w + hi * (1.0 - w)


@dataclass(slots=True, frozen=True)
class VasicekFactor:
    """Holder for a Vasicek single-factor parametrisation.

    Use either an explicit ``rho`` or a Basel-IRB-derived one (``rho=None``
    triggers :func:`basel_irb_correlation`).
    """

    pd: float
    lgd: float = 0.45
    rho: float | None = None
    maturity: float = 1.0
    asset_class: str = "corporate"

    def effective_rho(self) -> float:
        if self.rho is not None:
            return float(self.rho)
        return float(basel_irb_correlation(self.pd, asset_class=self.asset_class))

    def loss_cdf(self, x: ArrayLike) -> FloatArray:
        return vasicek_loss_cdf(x, self.pd, self.effective_rho())

    def var(self, alpha: float = 0.999) -> float:
        return float(vasicek_var(self.pd, self.effective_rho(), alpha=alpha))

    def capital(self, *, alpha: float = 0.999, apply_maturity_adjustment: bool = True) -> float:
        return float(
            basel_irb_capital(
                self.pd,
                self.lgd,
                self.effective_rho(),
                maturity=self.maturity,
                alpha=alpha,
                apply_maturity_adjustment=apply_maturity_adjustment,
                asset_class=self.asset_class,
            )
        )


__all__ = [
    "VasicekFactor",
    "basel_irb_capital",
    "basel_irb_correlation",
    "vasicek_loss_cdf",
    "vasicek_var",
]
