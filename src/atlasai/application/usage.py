from dataclasses import dataclass
from threading import Lock
from typing import Protocol

from atlasai.domain.models import TokenUsageSummary


@dataclass(frozen=True)
class UsageRecord:
    """Provider-reported token usage."""

    operation: str
    run_id: str | None = None
    provider: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    status: str = "known"
    trace_id: str | None = None


class UsageRepository(Protocol):
    """Usage record access."""

    def list_records(self, *, user_id: str) -> list[UsageRecord]: ...

    def append_record(self, *, user_id: str, record: UsageRecord) -> bool: ...


class InMemoryUsageRepository:
    """In-memory usage record store."""

    def __init__(self) -> None:
        self._records: dict[str, list[UsageRecord]] = {}
        self._seen_run_ids: dict[str, set[str]] = {}
        self._lock = Lock()

    def list_records(self, *, user_id: str) -> list[UsageRecord]:
        with self._lock:
            return list(self._records.get(user_id, []))

    def append_record(self, *, user_id: str, record: UsageRecord) -> bool:
        with self._lock:
            if record.run_id is not None:
                seen_run_ids = self._seen_run_ids.setdefault(user_id, set())
                if record.run_id in seen_run_ids:
                    return False
                seen_run_ids.add(record.run_id)
            self._records.setdefault(user_id, []).append(record)
            return True


class UsageService:
    """Reads and records token usage."""

    def __init__(self, repository: UsageRepository) -> None:
        self.repository = repository

    def get_token_summary(self, *, user_id: str) -> TokenUsageSummary:
        records = self.repository.list_records(user_id=user_id)
        input_total = 0
        output_total = 0
        total = 0

        for record in records:
            if record.status != "known":
                continue
            if record.input_tokens is not None:
                input_total += record.input_tokens
            if record.output_tokens is not None:
                output_total += record.output_tokens
            if record.total_tokens is not None:
                total += record.total_tokens

        return TokenUsageSummary(
            input=input_total,
            output=output_total,
            total=total,
        )

    def record_usage(self, *, user_id: str, record: UsageRecord) -> bool:
        return self.repository.append_record(user_id=user_id, record=record)
