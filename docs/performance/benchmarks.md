# Benchmarks

Reference timings from `pytest-benchmark` (Apple Silicon M-class laptop, 10
cores, Python 3.12, Numba 0.61). Re-run on your machine with:

```{code-block} bash
uv pip install -e ".[dev,bench]"
pytest tests/benchmarks --benchmark-only
```

## Single firm

| Operation                       | Mean       | Throughput  |
|---------------------------------|------------|-------------|
| `distance_to_default` (scalar)  | ~18 µs     | 55 kops/s   |
| `prob_of_default` (scalar)      | ~1.2 µs    | 800 kops/s  |
| `fit(naive)`                    | ~38 µs     | 26 kops/s   |
| `fit(jmr_iterative)`            | ~76 µs     | 13 kops/s   |
| `fit(vassalou_xing)` (snapshot) | ~83 µs     | 12 kops/s   |
| `MertonModel.{set,get}_params`  | ~290 ns    | 3.4 Mops/s  |

## Vectorised math

| Operation                              | Mean    | Throughput |
|----------------------------------------|---------|------------|
| `distance_to_default` on 1 000 000 inputs | 2.5 ms  | 400 Mops/s |

## Panel calibration (joblib threading, all cores)

| Panel size      | Method            | Mean    |
|-----------------|-------------------|---------|
| 1 000 firms     | naive             | ~0.05 s |
| 1 000 firms     | jmr_iterative     | ~0.3 s  |
| 10 000 firms    | naive             | ~0.5 s  |
| 10 000 firms    | jmr_iterative     | ~3 s    |
| 100 000 firms   | naive             | ~5 s    |
| 100 000 firms   | jmr_iterative     | ~30 s   |

## Calibration on a 252-day series

| Method                  | Mean    |
|-------------------------|---------|
| `vassalou_xing` (series)| ~200 ms |
| `duan_mle` (no surv. corr.) | ~600 ms |

## Portfolio Monte Carlo

| Workload                              | Mean |
|---------------------------------------|------|
| Vasicek analytic IRB, 5 000 firms     | ~4 ms |
| Basel IRB capital, 5 000 firms vector | ~0.5 ms |
| MC simulate, 5 000 firms × 100 000 sims | ~3 s  |

## Backtest metrics on 1 000 000 (PD, default) pairs

| Metric                | Mean   |
|-----------------------|--------|
| `auc`                 | ~50 ms |
| `brier`               | ~10 ms |
| `ks_statistic`        | ~70 ms |
| `roc_curve`           | ~80 ms |
| `calibration_curve`   | ~60 ms |
| `hosmer_lemeshow`     | ~120 ms |

These are all well within the v1.0 GA targets. CodSpeed compares each PR
against the previous main commit and flags regressions >5%.
