"""Tests for the AUC / Brier / KS / Hosmer-Lemeshow metrics."""

from __future__ import annotations

import numpy as np
import pytest

from merton.backtest import accuracy_ratio, auc, brier, hosmer_lemeshow, ks_statistic
from merton.exceptions import MertonInputError


def _independent_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Brute-force pairwise AUC for cross-checking."""
    pos = scores[y == 1]
    neg = scores[y == 0]
    s = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                s += 1.0
            elif p == n:
                s += 0.5
    return s / (pos.size * neg.size)


def _independent_brier(scores: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((scores - y) ** 2))


class TestAUC:
    def test_perfect_separation(self) -> None:
        y = np.array([0, 0, 0, 1, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert auc(s, y) == 1.0

    def test_random_predictor_around_half(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.3, 5_000)
        s = rng.uniform(0, 1, 5_000)
        v = auc(s, y)
        assert 0.45 < v < 0.55

    def test_matches_brute_force(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.3, 100).astype(float)
        s = rng.uniform(0, 1, 100)
        assert auc(s, y) == pytest.approx(_independent_auc(s, y), abs=1e-12)

    def test_accuracy_ratio_identity(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.binomial(1, 0.3, 500).astype(float)
        s = np.clip(y * 0.5 + rng.normal(0, 0.2, 500), 0, 1)
        assert accuracy_ratio(s, y) == pytest.approx(2.0 * auc(s, y) - 1)

    def test_undefined_one_class(self) -> None:
        with pytest.raises(MertonInputError):
            auc(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0]))

    def test_bad_labels_raise(self) -> None:
        with pytest.raises(MertonInputError):
            auc(np.array([0.1, 0.2]), np.array([0, 2]))


class TestBrier:
    def test_perfect_predictions(self) -> None:
        y = np.array([1.0, 0.0])
        s = np.array([1.0, 0.0])
        assert brier(s, y) == 0.0

    def test_matches_brute_force(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.3, 100).astype(float)
        s = rng.uniform(0, 1, 100)
        assert brier(s, y) == pytest.approx(_independent_brier(s, y))


class TestKS:
    def test_perfect_separation(self) -> None:
        y = np.array([0, 0, 0, 1, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert ks_statistic(s, y) == 1.0

    def test_random_predictor_low_ks(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.3, 5_000)
        s = rng.uniform(0, 1, 5_000)
        assert ks_statistic(s, y) < 0.15


class TestHosmerLemeshow:
    def test_returns_chi2_and_dof(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.3, 200).astype(float)
        s = np.clip(rng.uniform(0.1, 0.5, 200), 0, 1)
        chi2, dof = hosmer_lemeshow(s, y, bins=10)
        assert dof == 8
        assert np.isfinite(chi2)

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(MertonInputError):
            hosmer_lemeshow(np.array([0.1, 0.5]), np.array([0, 1]), bins=10)
