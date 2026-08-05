from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ThreadMessageResponse(BaseModel):
    """Persisted thread message."""

    message_id: str
    role: str
    content: str
    created_at: datetime
    assets: list[dict[str, str]] | None = None
    status: str = "done"


class ThreadResponse(BaseModel):
    """Thread response."""

    thread_id: str
    created_at: datetime
    expires_at: datetime
    document_id: str | None
    messages: list[ThreadMessageResponse] = []


class ThreadCreateRequest(BaseModel):
    """Thread creation request."""

    document_id: str | None = None


class ThreadMessageRequest(BaseModel):
    """Thread message request."""

    user_input: str


class StatusEvent(BaseModel):
    """Status stream event."""

    type: Literal["status"]
    text: str


class TokenEvent(BaseModel):
    """Token stream event."""

    type: Literal["token"]
    text: str


class UsageEvent(BaseModel):
    """Usage stream event."""

    type: Literal["usage"]
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    status: str


class DoneEvent(BaseModel):
    """Completion stream event."""

    type: Literal["done"]
    thread_id: str


class ErrorEvent(BaseModel):
    """Error stream event."""

    type: Literal["error"]
    text: str
