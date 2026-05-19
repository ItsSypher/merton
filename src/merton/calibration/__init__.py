"""Calibration methods for the Merton structural credit-risk model.

Each calibrator infers the unobserved asset value ``A`` and asset volatility
``σ_A`` from observable equity quantities (price level, equity volatility) and
the debt structure.

Available methods (v0.1):

- :func:`vassalou_xing` — iterative MLE (Vassalou & Xing, 2004). Default.
- :func:`jmr_iterative` — Jones, Mason & Rosenfeld (1984) two-equation system.
- :func:`naive` — Bharath & Shumway (2008) closed-form approximation. Fast.
"""

from __future__ import annotations

from .base import CalibrationResult, Calibrator, available_methods, get, register
from .jmr_iterative import JMRCalibrator, jmr_iterative
from .naive import NaiveCalibrator, naive
from .vassalou_xing import VassalouXingCalibrator, vassalou_xing

__all__ = [
    "CalibrationResult",
    "Calibrator",
    "JMRCalibrator",
    "NaiveCalibrator",
    "VassalouXingCalibrator",
    "available_methods",
    "get",
    "jmr_iterative",
    "naive",
    "register",
    "vassalou_xing",
]
