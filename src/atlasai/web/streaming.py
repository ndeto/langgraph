import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Literal

from atlasai.application.usage import UsageRecord

StreamFormat = Literal["text", "ndjson"]
NDJSON_KEEPALIVE_SECONDS = 14.0


async def stream_with_keepalive(
    source: AsyncIterator[str],
    *,
    interval_seconds: float = NDJSON_KEEPALIVE_SECONDS,
) -> AsyncIterator[str]:
    """Emit ignorable blank NDJSON lines while the source is silent."""

    iterator = source.__aiter__()
    pending = asyncio.ensure_future(anext(iterator))
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=interval_seconds)
            if not done:
                yield "\n"
                continue
            try:
                item = pending.result()
            except StopAsyncIteration:
                return
            yield item
            pending = asyncio.ensure_future(anext(iterator))
    finally:
        if not pending.done():
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending


def format_graph_chunk(chunk: object, stream_format: StreamFormat) -> str | None:
    """Format legacy graph stream chunks."""

    if not isinstance(chunk, dict):
        return None

    chunk_type = chunk.get("type")
    data = chunk.get("data")

    match stream_format:
        case "text":
            return format_text_chunk(chunk_type, data)
        case "ndjson":
            return format_legacy_ndjson_chunk(chunk_type, data)


def format_text_chunk(chunk_type: object, data: object) -> str | None:
    """Format plain text stream chunks."""

    match chunk_type:
        case "status" if isinstance(data, str):
            return f"{data}\n"
        case "token" if isinstance(data, str):
            return data
        case _:
            return None


def format_legacy_ndjson_chunk(chunk_type: object, data: object) -> str | None:
    """Format legacy NDJSON stream chunks."""

    match chunk_type:
        case "status" | "token" if isinstance(data, str):
            return encode_ndjson_event({"type": chunk_type, "text": data})
        case "rag_images" if isinstance(data, str):
            return encode_ndjson_event({"type": "rag_images", "markdown": data})
        case "final":
            return encode_ndjson_event({"type": "done"})
        case _:
            return None


def format_structured_ndjson_chunk(chunk: object) -> str | None:
    """Format versioned NDJSON stream chunks."""

    if not isinstance(chunk, dict):
        return None

    chunk_type = chunk.get("type")
    data = chunk.get("data")

    match chunk_type:
        case "status" | "token" if isinstance(data, str):
            return encode_ndjson_event({"type": chunk_type, "text": data})
        case "sources" if isinstance(data, list):
            return encode_ndjson_event({"type": "sources", "assets": data})
        case _:
            return None


def extract_usage_payload(final_state: object) -> object:
    """Extract the final provider payload."""

    if isinstance(final_state, dict):
        usage_payload = final_state.get("usage_payload")
        if usage_payload is not None:
            return usage_payload

        messages = final_state.get("messages")
        if isinstance(messages, list) and messages:
            for message in reversed(messages):
                usage_metadata = getattr(message, "usage_metadata", None)
                response_metadata = getattr(message, "response_metadata", None)
                if usage_metadata or response_metadata:
                    return message
            return messages[-1]
    return final_state


def build_usage_event(record: UsageRecord) -> dict[str, object]:
    """Build the public usage event."""

    return {
        "type": "usage",
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "total_tokens": record.total_tokens,
        "status": record.status,
    }


def encode_ndjson_event(event: dict[str, object]) -> str:
    """Encode one NDJSON event."""

    return json.dumps(event) + "\n"
