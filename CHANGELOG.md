# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (Phase 1.0 — Public launch)

- **First stable release**: dev-status classifier bumped to
  `5 - Production/Stable`; `merton.__all__` is the v1.0 API surface
  governed by [SemVer](https://semver.org/) (see
  `docs/contributing/api-stability.md`).
- **Lazy submodule loading**: `backtest`, `portfolio`, `reports`, and
  `batch_fit` are now resolved on first access via `__getattr__`. Cold
  `import merton` drops from ~1.1 s to ~500 ms on a 2024 M-series
  MacBook. `merton.calibration.covariance`, `merton.core.result`,
  `merton.core.term_structure`, and `merton._backend._numpy` defer
  scipy/pandas imports to first use.
- **conda-forge recipe** (`recipe/meta.yaml`) ready for the
  `conda-forge/staged-recipes` PR. Once merged, `conda install -c
  conda-forge merton` will be live.
- **Launch documentation**: `docs/blog/announcing-1.0.md` (release
  announcement), `docs/contributing/migrating-to-1.0.md` (migration
  guide for 0.x users).
- **`tests/performance/test_import_time.py`** locks the cold-import
  budget (`< 750 ms` on the dev workstation) so future PRs that
  re-introduce eager heavy imports fail CI.

### Added (Phase 0.9 — RC)

- **API stability surface**: `docs/contributing/api-stability.md` is the
  canonical reference for the v1.0 public API contract. Names in
  `merton.__all__` are stable from 1.0; underscore-prefixed modules are
  internal.
- **Deprecation helper** (`merton._deprecation`): `deprecated`,
  `deprecated_alias`, and `warn_deprecated` route renamed/retired public
  names through a `DeprecationWarning` with the removal target stamped in.
- **Security CI** (`.github/workflows/security.yml`): weekly `bandit`,
  `pip-audit`, and OSV scanner runs on `main`; also runs on PRs that
  touch dependencies or the security config.
- **Public-API surface tests** (`tests/unit/test_public_api.py`): every
  name in `merton.__all__` resolves; `__all__` is sorted and excludes
  private names; the package version is PEP 440.
- **Slow-test gating**: long MCMC tests are tagged `@pytest.mark.slow`
  and deselected from the default `pytest` invocation; full suite now
  runs in ~15 s. Run the slow tier with `pytest -m slow`.

### Changed (Phase 0.9 — RC)

- `merton.__init__.__getattr__` no longer advertises `io`, `diagnostics`,
  or `viz` — those namespaces are roadmapped for 1.x and their helpers
  currently live on `FirmPanel`, `MertonResult.summary`, and
  `merton.reports`.
- `merton.greeks.autodiff` enables `jax_enable_x64` at import so the
  autodiff Greeks match the closed-form values to single-precision
  tolerance; `equity_theta_ad` now follows the option-pricing convention
  (returns `-∂E/∂T`).
- Bandit suppressions added inline (`obs.py` teardown, `bootstrap.py`
  resample exception path, `cli/commands/excel.py` subprocess launcher).

### Added (Phase 0.8)

- **OpenTelemetry observability** (`merton.obs`): opt-in OTel tracing
  via `enable()` / `disable()` with `span()` context manager and
  `traced()` decorator. Auto-enable through `MERTON_OBS=1`; endpoint
  override via `MERTON_OTLP_ENDPOINT`; optional console mirror via
  `MERTON_OBS_CONSOLE=1`. Pulls in `opentelemetry-{api,sdk,exporter-otlp}`
  via the new `[obs]` extra.
- **Scenarios package** (`merton.scenarios`):
  - `Scenario` ABC + `CompositeScenario` (chain with `|`) +
    `ScenarioResult` audit record.
  - Atomic shocks `equity_shock`, `vol_shock`, `rate_shock`, `debt_shock`
    for ad-hoc stress.
  - `ClimateScenario` with carbon-price path + sectoral PD multipliers,
    pass-through, and chronic physical-risk parameters. `Sector` enum
    with default emission-intensity table; `carbon_price_curve`
    piecewise-linear helper; `carbon_price_to_writedown` standalone.
- **NGFS Phase V (2024) scenarios**
  (`merton.scenarios.predefined.ngfs`): `net_zero_2050`,
  `delayed_transition`, `current_policies`, `fragmented_world`.
- **ClimateOverlay structural model** (`merton.extensions.climate`):
  wraps any `StructuralModel` / `MertonModel` with a `ClimateScenario`
  + sector tag; writes down equity before calibration and scales PD by
  the sectoral multiplier.
- **Docs**: `docs/theory/scenarios.md` (general framework overview),
  `docs/theory/extensions/climate.md`,
  `docs/tutorials/02_climate_scenarios.md`,
  `docs/cookbook/observability.md`, `docs/cookbook/spark-dask.md`.
  Phase 0.8 features (`Scenario` framework, NGFS climate stress,
  OpenTelemetry) are now called out in the README Highlights block.

### Added (Phase 0.7)

- **CreditGrades model** (`merton.extensions.creditgrades`):
  Finger-Finkelstein-Pan-Lardy-Ta-Tierney 2002 closed-form survival /
  PD / implied CDS spread plus `CreditGradesModel.fit`. Random
  default-barrier widens short-horizon credit spreads vs Merton.
- **Leland-Toft endogenous-default model**
  (`merton.extensions.leland_toft`): optimal default boundary
  ``V_B^*``, PD, equity / debt values, and `LelandToftModel` calibrator
  on coupon-paying perpetual debt with taxes and bankruptcy costs.
- **Zhou / Merton jump-diffusion**
  (`merton.extensions.jump_diffusion`): Poisson-weighted series PD,
  Monte Carlo path simulator (`simulate_jump_diffusion`), and
  `JumpDiffusionModel` that calibrates ``(A, σ_A)`` via JMR and adds
  the jump contribution.
- **Longstaff-Schwartz two-factor model**
  (`merton.extensions.longstaff_schwartz`): Vasicek short-rate
  dynamics + asset GBM with correlation; closed-form PD at
  ``ρ = 0`` and vectorised Monte Carlo first-passage estimator
  otherwise.
- **Bayesian MCMC calibrator**
  (`merton.calibration.bayesian_mcmc`): `bayesian_mcmc` and
  `BayesianMCMCCalibrator` wrap `emcee.EnsembleSampler` around the
  Duan log-likelihood. Returns a `BayesianCalibrationResult` exposing
  the full posterior chain, log-probabilities, autocorrelation time,
  acceptance fraction, and percentile credible intervals.
- **Theory docs**: `docs/theory/extensions/{creditgrades, leland-toft,
  jump-diffusion, longstaff-schwartz}.md` and
  `docs/theory/bayesian-mcmc.md`.

### Added (Phase 0.6)

- **Excel integration via xlwings Server (FastAPI)**:
  - `merton.excel.functions` — pure-Python wrappers for every formula:
    `merton_dd`, `merton_pd`, `merton_spread`, `merton_asset_value`,
    `merton_asset_vol`, `merton_greeks`, `merton_pd_term`,
    `merton_backtest`, `merton_portfolio_var`, `merton_black_cox`.
  - `merton.excel.server` — FastAPI app exposing ``/healthz``,
    ``/functions.json``, ``/static/functions.{js,html}``,
    ``/taskpane.html``, and ``/call``. Lazy FastAPI import.
  - `merton.excel.manifest` — Office.js manifest XML generator with
    deterministic add-in UUID per base URL.
  - `merton.excel.installer` — write / remove the manifest in the OS's
    Excel sideload directory (macOS / Windows / Linux fallback).
- **Classic xlwings UDF fallback** (`merton.excel.udf`): registers
  every formula via the legacy `@xw.func` decorator for Windows
  desktop where running a local HTTP server isn't convenient.
- **CLI commands** (`merton excel ...`):
  - `install --url <BASE_URL>` writes the manifest.
  - `uninstall` removes the manifest.
  - `status` reports manifest + server PID.
  - `server start [--host --port --reload --background]` runs uvicorn.
  - `server stop` graceful SIGTERM via a PID file under
    `platformdirs.user_runtime_dir`.
  - `sample-workbook --out=<PATH>` writes a worked-example workbook.
- **Sample workbook** (`merton.excel.sample.write_sample_workbook`):
  programmatic openpyxl file with *Read me*, *Single firm*, *Portfolio*,
  *Backtest*, and *Function reference* sheets.
- **`python -m merton`** entrypoint via `src/merton/__main__.py`.
- **Docs**: `docs/excel/{installation, functions, sample-workbook}.md`,
  cookbook `excel-dashboard.md` with a full live-dashboard recipe.

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

[Unreleased]: https://github.com/ItsSypher/merton/compare/HEAD...HEAD
