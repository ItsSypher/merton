"""Property-based tests using Hypothesis.

These guarantee mathematical invariants of the Merton model independent of
any particular numerical input.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from merton import distance_to_default, prob_of_default
from merton.core.spread import implied_credit_spread

finite_floats_positive = st.floats(
    min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False
)
finite_floats_vol = st.floats(min_value=1e-3, max_value=2.0, allow_nan=False)
finite_floats_rate = st.floats(min_value=-0.05, max_value=0.2, allow_nan=False)
finite_floats_T = st.floats(min_value=1 / 365, max_value=30.0, allow_nan=False)


@given(
    A=finite_floats_positive,
    s=finite_floats_vol,
    D=finite_floats_positive,
    r=finite_floats_rate,
    T=finite_floats_T,
)
@settings(deadline=None, max_examples=200)
def test_pd_in_unit_interval(A: float, s: float, D: float, r: float, T: float) -> None:
    """PD = Φ(-d₂) must always live in [0, 1]."""
    dd = float(distance_to_default(A, s, D, r, T))
    pd = float(prob_of_default(dd))
    assert 0.0 <= pd <= 1.0


@given(
    A=finite_floats_positive,
    s1=finite_floats_vol,
    s2=finite_floats_vol,
    D=finite_floats_positive,
    r=finite_floats_rate,
    T=finite_floats_T,
)
@settings(deadline=None, max_examples=200)
def test_dd_monotone_decreasing_in_vol(
    A: float, s1: float, s2: float, D: float, r: float, T: float
) -> None:
    """Higher asset vol ⇒ lower DD, provided the firm is above its default
    threshold (otherwise the relationship can flip — a deeply underwater firm
    benefits from more variance via the option's gamma)."""
    assume(s1 < s2)
    # ∂d₂/∂σ < 0 iff A/D > exp(-σ²T/2 - rT). The threshold is decreasing in σ,
    # so the tightest binding is at σ₁ (the smaller value).
    threshold = float(np.exp(-0.5 * s1**2 * T - r * T))
    assume(threshold * 1.05 < A / D)
    dd_low = float(distance_to_default(A, s1, D, r, T))
    dd_high = float(distance_to_default(A, s2, D, r, T))
    assert dd_low > dd_high or np.isclose(dd_low, dd_high)


@given(
    A1=finite_floats_positive,
    A2=finite_floats_positive,
    s=finite_floats_vol,
    D=finite_floats_positive,
    r=finite_floats_rate,
    T=finite_floats_T,
)
@settings(deadline=None, max_examples=200)
def test_dd_monotone_increasing_in_assets(
    A1: float, A2: float, s: float, D: float, r: float, T: float
) -> None:
    """More assets ⇒ higher DD."""
    assume(A1 < A2)
    dd_low = float(distance_to_default(A1, s, D, r, T))
    dd_high = float(distance_to_default(A2, s, D, r, T))
    assert dd_high >= dd_low - 1e-9


@given(
    pd=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    T=finite_floats_T,
    lgd=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(deadline=None, max_examples=200)
def test_implied_spread_nonnegative(pd: float, T: float, lgd: float) -> None:
    """Spreads must be ≥ 0 for valid (PD, LGD) pairs."""
    s = float(implied_credit_spread(pd, T, lgd))
    assert s >= 0
    assert np.isfinite(s)
