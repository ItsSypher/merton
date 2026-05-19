"""Calibration methods for the Merton structural credit-risk model.

Each calibrator infers the unobserved asset value ``A`` and asset volatility
``σ_A`` from observable equity quantities (price level, equity volatility) and
the debt structure.

Available methods:

- :func:`vassalou_xing` — iterative MLE (Vassalou & Xing, 2004). Default.
- :func:`duan_mle` — transformed-data MLE with survivorship correction
  (Duan, 1994; Duan-Gauthier-Simonato-Zaanoun, 2004).
- :func:`kmv_iterative` — Crosbie-Bohn / Moody's KMV variant.
- :func:`jmr_iterative` — Jones, Mason & Rosenfeld (1984) two-equation system.
- :func:`naive` — Bharath & Shumway (2008) closed-form approximation. Fast.
"""

from __future__ import annotations

from .base import CalibrationResult, Calibrator, available_methods, get, register
from .bootstrap import BootstrapResult, block_bootstrap_calibration
from .covariance import (
    ConfInt,
    cov_from_hessian,
    delta_method,
    standard_errors,
    wald_ci,
)
from .duan_mle import DuanMLECalibrator, duan_mle
from .jmr_iterative import JMRCalibrator, jmr_iterative
from .kmv_iterative import KMVCalibrator, kmv_iterative
from .naive import NaiveCalibrator, naive
from .vassalou_xing import VassalouXingCalibrator, vassalou_xing

__all__ = [
    "BootstrapResult",
    "CalibrationResult",
    "Calibrator",
    "ConfInt",
    "DuanMLECalibrator",
    "JMRCalibrator",
    "KMVCalibrator",
    "NaiveCalibrator",
    "VassalouXingCalibrator",
    "available_methods",
    "block_bootstrap_calibration",
    "cov_from_hessian",
    "delta_method",
    "duan_mle",
    "get",
    "jmr_iterative",
    "kmv_iterative",
    "naive",
    "register",
    "standard_errors",
    "vassalou_xing",
    "wald_ci",
]
