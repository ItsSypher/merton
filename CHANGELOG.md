# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (Phase 0.5)

- **CuPy backend** (`merton._backend._cupy`): NVIDIA-GPU implementations of
  `d1_d2`, `equity_value`, `distance_to_default_kernel`,
  `prob_of_default_kernel`. Lazy-imported under `merton[gpu]`. Backend
  dispatch routes CuPy-array inputs to the GPU automatically.
- **MLX backend** (`merton._backend._mlx`): Apple Silicon Metal kernels
  via `mlx.core`. Normal CDF derived from `mx.erf`. Lazy-imported under
  `merton[mlx]`. Unified-memory model means zero-copy from NumPy.
- **AOT-warmed Numba cache**: `merton.warm_cache()` now exercises every
  `@njit` kernel with representative inputs. `wheels.yml` runs this in
  `CIBW_BEFORE_TEST` so the compiled `.nbi/.nbc` files ship inside the
  wheel — users pay zero first-call JIT cost.
- **100k-firm benchmark suite** (`tests/benchmarks/`): pytest-benchmark
  coverage of single-firm fits, 1k/10k/100k panels, calibration on a
  252-day series, portfolio Monte Carlo, and backtest metrics on
  1 000 000 (PD, default) pairs. Excluded from the default `pytest`
  run via `addopts = "--ignore=tests/benchmarks"`.
- **Free-threaded CI**: dedicated `cp313t` job in `.github/workflows/test.yml`
  that asserts `sys._is_gil_enabled() == False` and runs the full unit +
  property + golden test suites under GIL-free Python.
- **Performance docs**: `docs/performance/{benchmarks, backend-selection,
  apple-silicon, free-threaded}.md` and `docs/cookbook/large-panels.md`.

### Added (Phase 0.4)

- **Extensions**:
  - `merton.extensions.BlackCoxModel` and `black_cox_pd` — first-passage
    barrier model with constant or exponentially-decaying barrier
    (closed-form risk-neutral PD via the reflection principle).
  - `merton.extensions.GeskeModel` and `geske_equity_value` /
    `geske_pd` — 2-period compound-option pricing using the
    bivariate-normal CDF.
  - `merton.extensions.StructuralModel` / `StructuralResult` — shared
    ABC + result dataclass for all non-vanilla structural models.
- **Portfolio**:
  - `merton.portfolio.Portfolio` — container plus Monte Carlo +
    analytic-Vasicek engines. Accepts a list of `Firm` objects *or* a
    pre-computed PD vector.
  - `merton.portfolio.LossDistribution` — VaR, expected shortfall,
    economic capital, per-firm contribution decomposition.
  - `merton.portfolio.VasicekFactor` — Vasicek single-factor analytics.
  - `merton.portfolio.basel_irb_correlation` + `basel_irb_capital` —
    BCBS-prescribed asset correlation and IRB unexpected-loss capital
    with the standard maturity adjustment.
  - `merton.portfolio.copulas.GaussianCopula` and `TCopula` for
    correlated-default sampling.
  - `merton.portfolio.asset_correlation_from_equity` and
    `granularity_adjustment` / `hhi` / `effective_n`.
- **Backtest harness**:
  - `merton.backtest.{auc, accuracy_ratio, brier, ks_statistic,
    hosmer_lemeshow}` — implementations validated against sklearn's
    equivalents (no sklearn dependency).
  - `merton.backtest.ROCCurve` and `roc_curve`; `CalibrationCurve` and
    `calibration_curve` / `calibration_plot`.
  - `merton.backtest.rolling_window` — slide AUC/Brier/KS over a panel.
  - `merton.backtest.Backtest` + `BacktestResult` orchestrator with
    `add_metric`, `to_dict`, `summary`.
- **Reports**:
  - `merton.reports.render_backtest_report` — dependency-light
    standalone HTML report with embedded ROC + calibration SVGs.
- **Docs**: theory pages for Black-Cox, Geske, Vasicek, copulas, and
  metrics; cookbook recipes for `yfinance` and Bloomberg ingestion.
- **Testing**: extensive test coverage (336 tests, 90.87% coverage),
  with the gate held at 90% via `fail_under` in `pyproject.toml`.

### Added (Phase 0.3)

- JAX-backed kernels (`merton._backend._jax`): `jit`-compiled `d1_d2`,
  `equity_value`, `distance_to_default_kernel`, `prob_of_default_kernel`.
  Lazy-imported; only loaded when JAX is installed.
- Backend dispatch transparently routes JAX arrays to the JAX backend
  (zero-copy stay on device).
- JAX autodiff Greeks (`merton.greeks.autodiff`): `equity_delta_ad`,
  `equity_gamma_ad`, `equity_vega_ad`, `equity_theta_ad`, `equity_rho_ad`,
  `pd_leverage_sensitivity_ad`, `pd_vol_sensitivity_ad`,
  `pd_rate_sensitivity_ad`. Each is `jit`-compiled and `vmap`-friendly.
- `merton.FirmPanel` — Arrow-backed columnar container with constructors
  for pandas / polars / Arrow / CSV / Parquet / dict, columnar accessors,
  Firm-row iteration, boolean masks, and zero-copy slicing.
- `merton.batch_fit(panel_or_df, *, method, n_jobs, dispatch, progress,
  on_error, …)` — joblib-threaded panel calibration; returns the same
  dataframe type you handed in (pandas / polars / Arrow).
- `merton.batch.dispatch.parallel_map` — pluggable joblib / sequential /
  dask / ray dispatcher.
- `merton` CLI (typer): `merton --version`, `merton doctor` (Python build,
  GIL status, installed backends, GPU/MLX/JAX devices, dependency
  versions, suggested extras), `merton config show|set|reset`,
  `merton fit <input>` for single-firm and panel calibration.
- `MERTON_CONFIG_DIR` env var lets users / tests redirect the persisted
  config file location.
- Added `pyarrow>=15` and `tomli-w>=1.0` to core dependencies.
- Cookbook: `docs/cookbook/panel-fitting.md`, `docs/cookbook/jax-acceleration.md`.

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
