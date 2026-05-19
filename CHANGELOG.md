# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
