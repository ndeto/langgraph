from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class AnonymousSession:
    """Anonymous browser session state."""

    user_id: str
    expires_at: datetime
    active_document: str | None = None
    active_thread: str | None = None


@dataclass(frozen=True)
class QuotaBucket:
    """Quota totals for a single resource."""

    limit: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return max(self.limit - self.used, 0)


@dataclass(frozen=True)
class TokenUsageSummary:
    """Token usage totals."""

    input: int = 0
    output: int = 0
    total: int = 0


@dataclass(frozen=True)
class SessionQuotaSummary:
    """Session quota summary."""

    questions: QuotaBucket
    uploads: QuotaBucket
    tokens: TokenUsageSummary
