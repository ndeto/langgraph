from dataclasses import dataclass
from threading import Lock
from typing import Protocol

from atlasai.domain.models import QuotaBucket, SessionQuotaSummary, TokenUsageSummary


@dataclass(frozen=True)
class QuotaPolicy:
    """Quota limits for the demo application."""

    user_question_limit: int
    user_upload_limit: int
    ip_question_limit: int
    ip_upload_limit: int
    concurrent_ingestions_per_user: int
    concurrent_agent_runs_per_user: int


@dataclass(frozen=True)
class QuotaSnapshot:
    """Current quota counters for a user."""

    questions_used: int = 0
    uploads_used: int = 0
    active_ingestions: int = 0
    active_agent_runs: int = 0


@dataclass(frozen=True)
class RequestKeyQuotaSnapshot:
    """Current quota counters for a request key."""

    questions_used: int = 0
    uploads_used: int = 0


@dataclass(frozen=True)
class AdmissionResult:
    """Quota admission result."""

    allowed: bool
    reason: str | None = None


class QuotaRepository(Protocol):
    """Quota snapshot access."""

    def get_snapshot(self, *, user_id: str) -> QuotaSnapshot: ...

    def get_request_key_snapshot(
        self,
        *,
        request_key: str,
    ) -> RequestKeyQuotaSnapshot: ...

    def claim_question(
        self,
        *,
        user_id: str,
        request_key: str,
        user_limit: int,
        request_key_limit: int,
        concurrency_limit: int,
    ) -> AdmissionResult: ...

    def claim_upload(
        self,
        *,
        user_id: str,
        request_key: str,
        user_limit: int,
        request_key_limit: int,
        concurrency_limit: int,
    ) -> AdmissionResult: ...

    def release_agent_run(self, *, user_id: str) -> None: ...

    def release_ingestion(self, *, user_id: str) -> None: ...


class InMemoryQuotaRepository:
    """In-memory quota snapshot store."""

    def __init__(self) -> None:
        self._snapshots: dict[str, QuotaSnapshot] = {}
        self._request_key_snapshots: dict[str, RequestKeyQuotaSnapshot] = {}
        self._lock = Lock()

    def get_snapshot(self, *, user_id: str) -> QuotaSnapshot:
        return self._snapshots.get(user_id, QuotaSnapshot())

    def get_request_key_snapshot(
        self,
        *,
        request_key: str,
    ) -> RequestKeyQuotaSnapshot:
        return self._request_key_snapshots.get(request_key, RequestKeyQuotaSnapshot())

    def set_snapshot(self, *, user_id: str, snapshot: QuotaSnapshot) -> None:
        with self._lock:
            self._snapshots[user_id] = snapshot

    def set_request_key_snapshot(
        self,
        *,
        request_key: str,
        snapshot: RequestKeyQuotaSnapshot,
    ) -> None:
        with self._lock:
            self._request_key_snapshots[request_key] = snapshot

    def claim_question(
        self,
        *,
        user_id: str,
        request_key: str,
        user_limit: int,
        request_key_limit: int,
        concurrency_limit: int,
    ) -> AdmissionResult:
        with self._lock:
            user_snapshot = self._snapshots.get(user_id, QuotaSnapshot())
            request_key_snapshot = self._request_key_snapshots.get(
                request_key,
                RequestKeyQuotaSnapshot(),
            )

            if user_snapshot.questions_used >= user_limit:
                return AdmissionResult(
                    allowed=False,
                    reason="Question quota exceeded.",
                )
            if request_key_snapshot.questions_used >= request_key_limit:
                return AdmissionResult(
                    allowed=False,
                    reason="Question quota exceeded for this request key.",
                )
            if user_snapshot.active_agent_runs >= concurrency_limit:
                return AdmissionResult(
                    allowed=False,
                    reason="Another agent run is already in progress.",
                )

            self._snapshots[user_id] = QuotaSnapshot(
                questions_used=user_snapshot.questions_used + 1,
                uploads_used=user_snapshot.uploads_used,
                active_ingestions=user_snapshot.active_ingestions,
                active_agent_runs=user_snapshot.active_agent_runs + 1,
            )
            self._request_key_snapshots[request_key] = RequestKeyQuotaSnapshot(
                questions_used=request_key_snapshot.questions_used + 1,
                uploads_used=request_key_snapshot.uploads_used,
            )
            return AdmissionResult(allowed=True)

    def claim_upload(
        self,
        *,
        user_id: str,
        request_key: str,
        user_limit: int,
        request_key_limit: int,
        concurrency_limit: int,
    ) -> AdmissionResult:
        with self._lock:
            user_snapshot = self._snapshots.get(user_id, QuotaSnapshot())
            request_key_snapshot = self._request_key_snapshots.get(
                request_key,
                RequestKeyQuotaSnapshot(),
            )

            if user_snapshot.uploads_used >= user_limit:
                return AdmissionResult(
                    allowed=False,
                    reason="Upload quota exceeded.",
                )
            if request_key_snapshot.uploads_used >= request_key_limit:
                return AdmissionResult(
                    allowed=False,
                    reason="Upload quota exceeded for this request key.",
                )
            if user_snapshot.active_ingestions >= concurrency_limit:
                return AdmissionResult(
                    allowed=False,
                    reason="Another document ingestion is already in progress.",
                )

            self._snapshots[user_id] = QuotaSnapshot(
                questions_used=user_snapshot.questions_used,
                uploads_used=user_snapshot.uploads_used + 1,
                active_ingestions=user_snapshot.active_ingestions + 1,
                active_agent_runs=user_snapshot.active_agent_runs,
            )
            self._request_key_snapshots[request_key] = RequestKeyQuotaSnapshot(
                questions_used=request_key_snapshot.questions_used,
                uploads_used=request_key_snapshot.uploads_used + 1,
            )
            return AdmissionResult(allowed=True)

    def release_agent_run(self, *, user_id: str) -> None:
        with self._lock:
            snapshot = self._snapshots.get(user_id, QuotaSnapshot())
            self._snapshots[user_id] = QuotaSnapshot(
                questions_used=snapshot.questions_used,
                uploads_used=snapshot.uploads_used,
                active_ingestions=snapshot.active_ingestions,
                active_agent_runs=max(snapshot.active_agent_runs - 1, 0),
            )

    def release_ingestion(self, *, user_id: str) -> None:
        with self._lock:
            snapshot = self._snapshots.get(user_id, QuotaSnapshot())
            self._snapshots[user_id] = QuotaSnapshot(
                questions_used=snapshot.questions_used,
                uploads_used=snapshot.uploads_used,
                active_ingestions=max(snapshot.active_ingestions - 1, 0),
                active_agent_runs=snapshot.active_agent_runs,
            )


class QuotaService:
    """Builds quota summaries and admission snapshots."""

    def __init__(
        self,
        policy: QuotaPolicy,
        repository: QuotaRepository,
        request_key: str | None = None,
    ) -> None:
        self.policy = policy
        self.repository = repository
        self.request_key = request_key

    def claim_question(
        self,
        *,
        user_id: str,
        request_key: str | None = None,
    ) -> AdmissionResult:
        return self.repository.claim_question(
            user_id=user_id,
            request_key=self._resolve_request_key(
                user_id=user_id,
                request_key=request_key,
            ),
            user_limit=self.policy.user_question_limit,
            request_key_limit=self.policy.ip_question_limit,
            concurrency_limit=self.policy.concurrent_agent_runs_per_user,
        )

    def claim_upload(
        self,
        *,
        user_id: str,
        request_key: str | None = None,
    ) -> AdmissionResult:
        return self.repository.claim_upload(
            user_id=user_id,
            request_key=self._resolve_request_key(
                user_id=user_id,
                request_key=request_key,
            ),
            user_limit=self.policy.user_upload_limit,
            request_key_limit=self.policy.ip_upload_limit,
            concurrency_limit=self.policy.concurrent_ingestions_per_user,
        )

    def release_agent_run(self, *, user_id: str) -> None:
        self.repository.release_agent_run(user_id=user_id)

    def release_ingestion(self, *, user_id: str) -> None:
        self.repository.release_ingestion(user_id=user_id)

    def get_session_quota_summary(
        self,
        *,
        user_id: str,
        token_usage: TokenUsageSummary,
    ) -> SessionQuotaSummary:
        snapshot = self.repository.get_snapshot(user_id=user_id)
        return SessionQuotaSummary(
            questions=QuotaBucket(
                limit=self.policy.user_question_limit,
                used=snapshot.questions_used,
            ),
            uploads=QuotaBucket(
                limit=self.policy.user_upload_limit,
                used=snapshot.uploads_used,
            ),
            tokens=token_usage,
        )

    def _resolve_request_key(self, *, user_id: str, request_key: str | None) -> str:
        return request_key or self.request_key or user_id
