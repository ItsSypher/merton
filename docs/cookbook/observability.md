# OpenTelemetry observability

`merton.obs` integrates the package with any OpenTelemetry-compatible tracing
backend (Datadog, Honeycomb, Grafana Tempo, Jaeger, Lightstep, …). Tracing
is **off by default** — wiring it up adds zero overhead to the hot path
until you call `merton.obs.enable(...)`.

## Installation

```bash
pip install "merton[obs]"
```

This pulls in `opentelemetry-api`, `opentelemetry-sdk`, and the OTLP gRPC
exporter.

## Enabling tracing

```python
import merton

merton.obs.enable(
    service_name="risk-engine",
    otlp_endpoint="http://otel-collector:4317",
)
```

After `enable(...)` returns, every public calibration entry-point that uses
the `@traced` decorator (Phase 0.9 will land the full coverage; the current
release exposes `obs.span(...)` and `obs.traced(...)` for you to wire into
your own code paths) emits a span on each call.

### Environment variables

| Variable | Effect |
|---|---|
| `MERTON_OBS=1` | Auto-enable on import using defaults. |
| `MERTON_OTLP_ENDPOINT=…` | Override the OTLP gRPC endpoint. |
| `MERTON_OBS_CONSOLE=1` | Mirror spans to stdout (helpful for debugging). |

## Adding your own spans

The `span` context manager and `traced` decorator give you full control
inside your own functions:

```python
from merton import obs

@obs.traced("portfolio.fit", n_firms=500)
def fit_portfolio(panel):
    with obs.span("merton.batch", method="duan_mle"):
        return panel.fit(method="duan_mle")
```

When tracing is off, both helpers are essentially free — a single
`is_enabled()` check costs well under a microsecond.

## Disabling

```python
merton.obs.disable()
```

`disable()` flushes any in-flight spans and shuts the provider down. Future
calls become no-ops again.

## Compatibility

`merton.obs` uses the standard OpenTelemetry Python SDK. Anything that
speaks OTLP gRPC (the OpenTelemetry Collector, OpenTelemetry-aware vendors,
or a sidecar like the Grafana Agent) can ingest the spans without further
configuration.
