"""Replicate Vassalou-Xing (2004) JMR-style calibrations against an
independent two-equation solver. The reference asset_value/σ_A are computed
inside the test so the comparison is definition vs implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import fsolve
from scipy.stats import norm

from merton.calibration import jmr_iterative

REF = Path(__file__).parent.parent / "data" / "reference" / "vassalou_xing_2004_table2.csv"


def _independent_jmr(row: pd.Series) -> tuple[float, float]:
    """Independent JMR two-equation solver using only scipy primitives."""
    E, sigma_E = float(row["equity"]), float(row["equity_vol"])
    D, r, T = float(row["debt"]), float(row["rf"]), float(row["T"])

    def residuals(x):
        A, sigma_A = x
        if A <= 0 or sigma_A <= 0:
            return [1e6, 1e6]
        d1 = (np.log(A / D) + (r + 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
        d2 = d1 - sigma_A * np.sqrt(T)
        eq1 = A * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
        eq2 = (A / E) * norm.cdf(d1) * sigma_A - sigma_E
        return [eq1, eq2]

    x0 = [E + D, sigma_E * E / (E + D)]
    sol, _, ier, _ = fsolve(residuals, x0, xtol=1e-12, full_output=True)
    assert ier == 1, f"reference solver did not converge for {row['firm']}"
    return float(sol[0]), float(sol[1])


@pytest.fixture(scope="module")
def cases() -> pd.DataFrame:
    return pd.read_csv(REF)


@pytest.mark.golden
class TestVassalouXing2004:
    def test_jmr_matches_independent_reference(self, cases: pd.DataFrame) -> None:
        for _, row in cases.iterrows():
            A_ref, sigma_ref = _independent_jmr(row)
            res = jmr_iterative(
                equity=float(row["equity"]),
                equity_vol=float(row["equity_vol"]),
                debt=float(row["debt"]),
                rf=float(row["rf"]),
                T=float(row["T"]),
            )
            np.testing.assert_allclose(
                res.asset_value,
                A_ref,
                rtol=1e-5,
                err_msg=f"firm {row['firm']!r} asset_value",
            )
            np.testing.assert_allclose(
                res.asset_vol,
                sigma_ref,
                rtol=1e-5,
                err_msg=f"firm {row['firm']!r} asset_vol",
            )

    def test_implied_equity_matches_input(self, cases: pd.DataFrame) -> None:
        """The classic JMR consistency check: plug (A, σ_A) back into BSM
        and the implied equity must equal the input."""
        from merton import equity_value

        for _, row in cases.iterrows():
            res = jmr_iterative(
                equity=float(row["equity"]),
                equity_vol=float(row["equity_vol"]),
                debt=float(row["debt"]),
                rf=float(row["rf"]),
                T=float(row["T"]),
            )
            E_implied = float(
                equity_value(
                    res.asset_value,
                    res.asset_vol,
                    float(row["debt"]),
                    float(row["rf"]),
                    float(row["T"]),
                )
            )
            assert E_implied == pytest.approx(float(row["equity"]), rel=1e-6)
