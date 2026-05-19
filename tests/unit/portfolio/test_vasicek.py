"""Tests for the Vasicek single-factor analytics and Basel IRB capital."""

from __future__ import annotations

import numpy as np
import pytest

from merton.exceptions import MertonInputError
from merton.portfolio import (
    VasicekFactor,
    basel_irb_capital,
    basel_irb_correlation,
    vasicek_loss_cdf,
    vasicek_var,
)


class TestVasicekLossCDF:
    def test_at_pd_quantile(self) -> None:
        # F_L(PD) > 0 but small.
        pd = 0.05
        rho = 0.20
        v = float(vasicek_loss_cdf(pd, pd, rho))
        # By construction the median of the loss distribution exceeds PD
        # only when ρ > 0; this returns the CDF value at x = PD.
        assert 0 < v < 1

    def test_monotone_in_x(self) -> None:
        pd, rho = 0.05, 0.20
        xs = np.linspace(0.01, 0.5, 20)
        cdfs = vasicek_loss_cdf(xs, pd, rho)
        assert np.all(np.diff(cdfs) >= -1e-12)

    def test_rejects_invalid_inputs(self) -> None:
        with pytest.raises(MertonInputError):
            vasicek_loss_cdf(0.5, 1.5, 0.20)  # pd > 1
        with pytest.raises(MertonInputError):
            vasicek_loss_cdf(0.5, 0.05, -0.1)  # ρ < 0


class TestVasicekVaR:
    def test_var_increases_in_alpha(self) -> None:
        var_95 = float(vasicek_var(0.05, 0.2, alpha=0.95))
        var_99 = float(vasicek_var(0.05, 0.2, alpha=0.99))
        var_999 = float(vasicek_var(0.05, 0.2, alpha=0.999))
        assert var_95 < var_99 < var_999

    def test_var_increases_in_rho(self) -> None:
        var_low = float(vasicek_var(0.05, 0.05, alpha=0.999))
        var_hi = float(vasicek_var(0.05, 0.40, alpha=0.999))
        assert var_hi > var_low

    def test_rejects_bad_alpha(self) -> None:
        with pytest.raises(MertonInputError):
            vasicek_var(0.05, 0.2, alpha=1.5)


class TestBaselIRB:
    def test_correlation_matches_published_table_corporate(self) -> None:
        # BCBS-published bounds: ρ ranges from 0.12 (PD = 1) to 0.24 (PD → 0).
        # PD=0.0001 sits very close to the upper bound (w ≈ 0.005) ⇒ ρ ≈ 0.2394.
        rho_low_pd = float(basel_irb_correlation(0.0001))
        rho_high_pd = float(basel_irb_correlation(0.99))
        assert rho_low_pd == pytest.approx(0.2394, abs=1e-3)
        assert rho_high_pd == pytest.approx(0.12, abs=1e-3)

    def test_correlation_decreasing_in_pd(self) -> None:
        pds = np.array([0.001, 0.01, 0.05, 0.1, 0.3])
        rhos = basel_irb_correlation(pds)
        assert np.all(np.diff(rhos) < 0)

    def test_retail_lower_correlation(self) -> None:
        # Retail correlations are mechanically lower than corporate.
        assert float(basel_irb_correlation(0.05, asset_class="retail")) < float(
            basel_irb_correlation(0.05, asset_class="corporate")
        )

    def test_qrre_constant(self) -> None:
        # Qualifying revolving retail exposure correlation is a constant 4%.
        assert float(basel_irb_correlation(0.01, asset_class="qrre")) == pytest.approx(0.04)
        assert float(basel_irb_correlation(0.50, asset_class="qrre")) == pytest.approx(0.04)

    def test_unknown_asset_class_raises(self) -> None:
        with pytest.raises(MertonInputError):
            basel_irb_correlation(0.05, asset_class="cars")


class TestBaselCapital:
    def test_capital_positive_and_below_lgd(self) -> None:
        k = float(basel_irb_capital(0.01, 0.45, maturity=2.5))
        assert 0 < k < 0.45

    def test_capital_increases_with_pd_below_25pct(self) -> None:
        ks = [
            float(basel_irb_capital(p, 0.45, maturity=2.5)) for p in [0.001, 0.01, 0.05, 0.10, 0.25]
        ]
        # Monotone over the realistic range.
        assert all(ks[i] < ks[i + 1] for i in range(len(ks) - 1))

    def test_capital_without_maturity_adjustment(self) -> None:
        k_with = float(basel_irb_capital(0.01, 0.45, maturity=5.0, apply_maturity_adjustment=True))
        k_no = float(basel_irb_capital(0.01, 0.45, apply_maturity_adjustment=False))
        # Maturity > 1 boosts capital with the adjustment on.
        assert k_with > k_no


class TestVasicekFactorClass:
    def test_basic(self) -> None:
        vf = VasicekFactor(pd=0.02, lgd=0.45, maturity=2.5)
        assert vf.effective_rho() > 0
        assert vf.var() > 0
        assert vf.capital() > 0

    def test_explicit_rho(self) -> None:
        vf = VasicekFactor(pd=0.02, lgd=0.45, rho=0.15)
        assert vf.effective_rho() == 0.15
