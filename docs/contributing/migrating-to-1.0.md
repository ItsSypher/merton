# Migrating to 1.0

The 1.0 release freezes the public API documented in
{doc}`api-stability`. If you've been tracking `merton` through the 0.x
series, this page is the punch list of what to verify in your code.

## TL;DR

The 0.9 RC kept the API stable through the freeze, so **most users won't
have any changes to make.** If your code only calls `Firm`, `fit`,
`MertonModel`, the calibrators, the extensions, the portfolio engine,
the backtest harness, the Excel UDFs, or the climate scenarios, you can
upgrade with a plain `pip install --upgrade merton` and a green test
suite.

## What changed since 0.x

### `merton.io`, `merton.diagnostics`, `merton.viz` are not part of 1.0

These three namespaces were planned in the original master plan but
never landed as standalone surfaces. They've been removed from
`merton.__all__` and `__getattr__` so the public surface is honest about
what's actually shipped. Their helpers live elsewhere:

| Old plan | Where it lives in 1.0 |
|---|---|
| `merton.io.from_csv` / `to_parquet` | `merton.FirmPanel.from_csv` / `.to_parquet` |
| `merton.diagnostics.summary` | `MertonResult.summary()` |
| `merton.viz.*` | `merton.reports.html` (HTML reports) |

If you were importing from these namespaces in a 0.x release, those
imports were always `ImportError`s — there's nothing in your code to
change beyond ignoring the old roadmap notes.

### `merton.batch_fit` is lazily loaded

`batch_fit` now imports pandas only when first called, so cold `import
merton` is ~500 ms instead of ~1.1 s. The function signature is
identical; no caller code needs to change.

### `merton.obs` is new in 0.8

The OpenTelemetry shim landed in Phase 0.8. Calling
`merton.obs.enable(...)` is opt-in and a no-op by default. If you
weren't using OTel before, this changes nothing.

### `equity_theta_ad` sign convention

Under `merton.greeks.autodiff`, `equity_theta_ad` now returns
`-∂E/∂T` to match the option-pricing convention used by the closed-form
`merton.greeks.equity_theta`. If your code was multiplying the autodiff
theta by -1 to align it, drop that adjustment.

This only affects users of `merton[jax]`; the closed-form Greeks haven't
changed sign.

### Deprecation policy

Any future public-name renames go through `merton._deprecation.deprecated`
and emit a `DeprecationWarning` with the removal version stamped in. The
package guarantees at least one full minor release before any name
disappears.

## Recommended upgrade sequence

```bash
# 1. Upgrade
uv pip install --upgrade merton

# 2. Run your test suite under -W error::DeprecationWarning
pytest -W error::DeprecationWarning

# 3. If any DeprecationWarning fires, follow the cited migration path.
```

If you hit a surprise, please open an issue.
