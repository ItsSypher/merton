# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (Phase 0.2)

- Duan (1994) transformed-data MLE calibrator (`merton.calibration.duan_mle`).
- Survivorship-bias correction via closed-form first-passage probability
  for geometric Brownian motion (`merton._backend._survival`).
- KMV / Crosbie-Bohn iterative calibrator (`merton.calibration.kmv_iterative`)
  with hookable empirical `edf_map`.
- MLE asymptotic standard errors, Wald confidence intervals, and a generic
  delta-method propagator (`merton.calibration.covariance`).
- Block-bootstrap CIs for time-series calibrators
  (`merton.calibration.block_bootstrap_calibration`); wired into
  `MertonModel(n_bootstrap=…)`.
- `MertonResult.confidence_interval(level, method)` lazy method (asymptotic
  or bootstrap).
- `MertonResult` now exposes `dd_series` / `pd_series` / `asset_value_series`
  for time-series calibrations; the scalar `dd` / `pd` are the most-recent
  observation.
- Paper-replication test suite (`tests/golden/`) covering Bharath-Shumway
  2008, Vassalou-Xing 2004, and a synthetic Duan MLE recovery test.
- Executable AAPL-style tutorial (`docs/tutorials/01_single_firm_aapl.md`).

### Added (Phase 0.1)

- Initial release scaffolding (pyproject.toml, CI, docs skeleton).
- Core single-firm Merton model: `Firm`, `MertonModel`, `MertonResult`.
- Calibration methods: Vassalou-Xing iterative MLE, Jones-Mason-Rosenfeld
  iterative, Bharath-Shumway naive.
- Vectorized math primitives: `distance_to_default`, `prob_of_default`,
  `implied_credit_spread`, `physical_pd`, `term_structure_pd`.
- Closed-form equity Greeks: delta, gamma, vega, theta, rho.
- PD sensitivities to leverage, asset volatility, and risk-free rate.
- Default-point formulas: KMV (ST + 0.5·LT), total debt, short-only, custom.
- NumPy backend (default) with backend-dispatch infrastructure ready for
  Numba, CuPy, JAX, and MLX in subsequent phases.

[Unreleased]: https://github.com/merton-credit/merton/compare/HEAD...HEAD
