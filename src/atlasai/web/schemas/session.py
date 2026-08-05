from datetime import datetime

from pydantic import BaseModel


class QuotaBucketResponse(BaseModel):
    """Quota bucket response."""

    limit: int
    used: int
    remaining: int


class TokenUsageResponse(BaseModel):
    """Token usage response."""

    input: int
    output: int
    total: int


class SessionQuotaResponse(BaseModel):
    """Session quota response."""

    questions: QuotaBucketResponse
    uploads: QuotaBucketResponse
    tokens: TokenUsageResponse


class SessionResponse(BaseModel):
    """Anonymous session response."""

    user_id: str
    expires_at: datetime
    active_document: str | None
    active_thread: str | None
    quota: SessionQuotaResponse
