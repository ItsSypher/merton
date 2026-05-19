"""OpenTelemetry observability for merton.

Wraps the heavy calibration / panel / portfolio entry points with
OpenTelemetry spans so production users get end-to-end traces in their
existing OTel pipeline (Datadog, Honeycomb, Grafana Tempo, …).

The module is **lazy** — calling :func:`enable` is what wires up the
TracerProvider. Before that, :func:`traced` is a no-op decorator, so
``import merton`` carries zero observability overhead for users who
haven't opted in.

Install with::

    pip install "merton[obs]"

then::

    import merton

    merton.obs.enable(service_name="risk-engine", otlp_endpoint="http://otel-collector:4317")
    # All MertonModel.fit / batch_fit / calibrator calls now emit spans.

Configuration via environment variables:

- ``MERTON_OBS=1`` — auto-enable on import (uses defaults).
- ``MERTON_OTLP_ENDPOINT`` — override the default OTLP gRPC endpoint.
- ``MERTON_OBS_CONSOLE=1`` — also export to stdout (debugging).
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, TypeVar

from . import __version__

F = TypeVar("F", bound=Callable[..., Any])

_DEFAULT_OTLP_ENDPOINT = "http://localhost:4317"
_DEFAULT_SERVICE_NAME = "merton"

_state: dict[str, Any] = {
    "enabled": False,
    "tracer_provider": None,
    "tracer": None,
}


def is_enabled() -> bool:
    """Return ``True`` if OTel tracing is wired up in this process."""
    return bool(_state["enabled"])


def enable(
    *,
    service_name: str = _DEFAULT_SERVICE_NAME,
    otlp_endpoint: str | None = None,
    console: bool = False,
    insecure: bool = True,
) -> None:
    """Initialise the OpenTelemetry TracerProvider.

    Idempotent — calling multiple times keeps the first configuration.
    Raises :class:`ImportError` if ``merton[obs]`` isn't installed.
    """
    if _state["enabled"]:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as err:  # pragma: no cover - extras gate
        raise ImportError('merton.obs requires merton[obs]: pip install "merton[obs]"') from err

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": __version__,
            "library.name": "merton",
        }
    )
    provider = TracerProvider(resource=resource)

    endpoint = otlp_endpoint or os.environ.get("MERTON_OTLP_ENDPOINT", _DEFAULT_OTLP_ENDPOINT)
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=endpoint, insecure=insecure),
                max_queue_size=4096,
                max_export_batch_size=512,
            )
        )
    except ImportError:  # pragma: no cover - sub-extra missing
        # OTLP exporter is an optional dependency on top of the API; fall back
        # to the no-op exporter so users without it still see local console
        # output if requested.
        pass

    if console or os.environ.get("MERTON_OBS_CONSOLE") == "1":
        from opentelemetry.sdk.trace.export import (
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _state["tracer_provider"] = provider
    _state["tracer"] = trace.get_tracer("merton", __version__)
    _state["enabled"] = True


def disable() -> None:
    """Shut the current TracerProvider down. Tracing becomes a no-op again."""
    if not _state["enabled"]:
        return
    provider = _state.get("tracer_provider")
    if provider is not None:
        try:
            provider.force_flush()
            provider.shutdown()
        except Exception:
            pass
    _state["enabled"] = False
    _state["tracer"] = None
    _state["tracer_provider"] = None


def get_tracer():  # type: ignore[no-untyped-def]
    """Return the active tracer if enabled; ``None`` otherwise."""
    return _state["tracer"]


@contextmanager
def span(name: str, **attributes: Any):  # type: ignore[no-untyped-def]
    """Context manager that opens a span when tracing is on.

    No-op (yields ``None``) otherwise. Use this around any block of work
    you want measured::

        from merton import obs

        with obs.span("merton.fit", method="duan_mle"):
            result = model.fit(firm)
    """
    if not _state["enabled"]:
        yield None
        return
    tracer = _state["tracer"]
    with tracer.start_as_current_span(name, attributes=attributes) as current:
        try:
            yield current
        except Exception as exc:
            from opentelemetry.trace import Status, StatusCode

            current.record_exception(exc)
            current.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def traced(name: str | None = None, **default_attributes: Any) -> Callable[[F], F]:
    """Decorator that wraps a function in a span when tracing is enabled.

    The decorated function pays a single ``is_enabled()`` check per call
    when tracing is off — well under a microsecond.
    """

    def decorator(fn: F) -> F:
        span_name = name or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _state["enabled"]:
                return fn(*args, **kwargs)
            with span(span_name, **default_attributes):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


# Auto-enable from the environment if the user opted in.
if os.environ.get("MERTON_OBS") == "1":  # pragma: no cover - env-driven
    import contextlib

    with contextlib.suppress(ImportError):
        enable()


__all__ = [
    "disable",
    "enable",
    "get_tracer",
    "is_enabled",
    "span",
    "traced",
]
