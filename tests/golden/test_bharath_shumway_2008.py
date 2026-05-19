"""Replicate Bharath-Shumway (2008) naive DD/PD numbers from first principles.

The naive Bharath-Shumway estimator (Bharath & Shumway 2008, §3.2):

::

    A   = E + D
    σ_D = 0.05 + 0.25 · σ_E
    σ_A = (E/A)·σ_E + (D/A)·σ_D
    DD  = [ln(A/D) + (r - σ_A²/2)·T] / (σ_A·√T)
    PD  = Φ(-DD)

The reference CSV holds only the inputs; the expected values are recomputed
in the test from these formulas so the comparison is *definition vs
implementation*.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from merton.calibration import naive

REF = Path(__file__).parent.parent / "data" / "reference" / "bharath_shumway_2008.csv"


def _independent_naive(row: pd.Series) -> tuple[float, float, float, float]:
    """Reference implementation (no merton calls). Returns A, σ_A, DD, PD."""
    E, sigma_E = float(row["equity"]), float(row["equity_vol"])
    D, r, T = float(row["debt"]), float(row["rf"]), float(row["T"])
    A = E + D
    sigma_D = 0.05 + 0.25 * sigma_E
    sigma_A = (E / A) * sigma_E + (D / A) * sigma_D
    dd = (np.log(A / D) + (r - 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
    pd = float(norm.cdf(-dd))
    return float(A), float(sigma_A), float(dd), pd


@pytest.fixture(scope="module")
def cases() -> pd.DataFrame:
    return pd.read_csv(REF)


@pytest.mark.golden
class TestBharathShumway2008:
    def test_naive_calibrator_matches_definition(self, cases: pd.DataFrame) -> None:
        for _, row in cases.iterrows():
            A_ref, sigma_ref, _, _ = _independent_naive(row)
            res = naive(
                equity=row["equity"],
                equity_vol=row["equity_vol"],
                debt=row["debt"],
                rf=row["rf"],
                T=row["T"],
            )
            np.testing.assert_allclose(res.asset_value, A_ref, rtol=1e-12)
            np.testing.assert_allclose(res.asset_vol, sigma_ref, rtol=1e-12)

    def test_dd_pd_match_reference(self, cases: pd.DataFrame) -> None:
        from merton import distance_to_default, prob_of_default

        for _, row in cases.iterrows():
            A_ref, sigma_ref, dd_ref, pd_ref = _independent_naive(row)
            dd = float(
                distance_to_default(
                    asset_value=A_ref,
                    asset_vol=sigma_ref,
                    debt=float(row["debt"]),
                    rf=float(row["rf"]),
                    T=float(row["T"]),
                )
            )
            pd = float(prob_of_default(dd))
            assert dd == pytest.approx(dd_ref, rel=1e-7)
            assert pd == pytest.approx(pd_ref, rel=1e-6, abs=1e-12)
