"""Shared request and execution contracts for graph-backed services."""

from collections.abc import AsyncIterator
from typing import Protocol

from typing_extensions import NotRequired, TypedDict


class InvokePayload(TypedDict):
    """Trusted input required to execute one graph turn."""

    user_input: str
    thread_id: str
    user_id: NotRequired[str]
    document_id: NotRequired[str | None]
    trace_id: NotRequired[str]


class GraphRunner(Protocol):
    """Streaming interface for graph-based request execution."""

    def stream(
        self,
        payload: InvokePayload,
    ) -> AsyncIterator[object]:
        """Stream events produced while executing one graph turn."""

        ...
