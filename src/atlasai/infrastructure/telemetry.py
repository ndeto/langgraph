from collections.abc import Mapping
from dataclasses import dataclass
from uuid import uuid4

from atlasai.application.usage import UsageRecord


@dataclass(frozen=True)
class TelemetryContext:
    """Pseudonymous telemetry metadata."""

    operation: str
    user_id: str | None = None
    thread_id: str | None = None
    document_id: str | None = None
    trace_id: str | None = None
    provider: str | None = None
    model: str | None = None
    result: str | None = None


def build_telemetry_metadata(context: TelemetryContext) -> dict[str, str]:
    """Build stable telemetry metadata."""

    metadata = {
        "operation": context.operation,
        "user_id": context.user_id,
        "thread_id": context.thread_id,
        "document_id": context.document_id,
        "trace_id": context.trace_id,
        "provider": context.provider,
        "model": context.model,
        "result": context.result,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def ensure_trace_id(trace_id: str | None) -> str:
    """Return an existing trace ID or generate a new one."""

    return trace_id or str(uuid4())


def usage_record_from_callback(
    *,
    operation: str,
    payload: object | None = None,
    provider: str | None = None,
    model: str | None = None,
    trace_id: str | None = None,
    run_id: str | None = None,
) -> UsageRecord:
    """Build a usage record from callback metadata."""

    usage_metadata = _extract_usage_mapping(payload)
    input_tokens = _extract_int(usage_metadata, "input_tokens", "prompt_tokens")
    output_tokens = _extract_int(usage_metadata, "output_tokens", "completion_tokens")
    total_tokens = _extract_int(usage_metadata, "total_tokens")
    status = (
        "known"
        if any(
            token_count is not None
            for token_count in (input_tokens, output_tokens, total_tokens)
        )
        else "unknown"
    )

    return UsageRecord(
        operation=operation,
        run_id=run_id or _extract_run_id(payload),
        provider=provider or _extract_string(payload, "provider"),
        model=model or _extract_model(payload),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        status=status,
        trace_id=trace_id or _extract_string(payload, "trace_id"),
    )


def _extract_usage_mapping(payload: object | None) -> Mapping[str, object]:
    for candidate in (
        _lookup_value(payload, "usage_metadata"),
        _lookup_value(payload, "token_usage"),
        _lookup_value(_lookup_value(payload, "response_metadata"), "token_usage"),
        _lookup_value(_lookup_value(payload, "llm_output"), "token_usage"),
    ):
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _extract_model(payload: object | None) -> str | None:
    for candidate in (
        _extract_string(payload, "model"),
        _extract_string(payload, "model_name"),
        _extract_string(_lookup_value(payload, "response_metadata"), "model_name"),
        _extract_string(_lookup_value(payload, "response_metadata"), "model"),
    ):
        if candidate is not None:
            return candidate
    return None


def _extract_run_id(payload: object | None) -> str | None:
    for candidate in (
        _extract_string(payload, "run_id"),
        _extract_string(payload, "id"),
        _extract_string(_lookup_value(payload, "response_metadata"), "id"),
        _extract_string(_lookup_value(payload, "llm_output"), "run_id"),
    ):
        if candidate is not None:
            return candidate
    return None


def _extract_int(source: object | None, *keys: str) -> int | None:
    for key in keys:
        value = _lookup_value(source, key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
    return None


def _extract_string(source: object | None, key: str) -> str | None:
    value = _lookup_value(source, key)
    return value if isinstance(value, str) else None


def _lookup_value(source: object | None, key: str) -> object | None:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)
