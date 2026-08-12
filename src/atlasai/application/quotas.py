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
class BucketQuotaSnapshot:
    """Current quota counters for a quota bucket."""

    questions_used: int = 0
    uploads_used: int = 0


@dataclass(frozen=True)
class AdmissionResult:
    """Quota admission result."""

    allowed: bool
    code: str | None = None
    message: str | None = None

    @property
    def reason(self) -> str | None:
        return self.message


class QuotaRepository(Protocol):
    """Quota snapshot access."""

    def get_snapshot(self, *, user_id: str) -> QuotaSnapshot: ...

    def get_client_snapshot(
        self,
        *,
        client_key: str,
    ) -> BucketQuotaSnapshot: ...

    def get_ip_snapshot(
        self,
        *,
        ip_hash: str,
    ) -> BucketQuotaSnapshot: ...

    def claim_question(
        self,
        *,
        user_id: str,
        client_key: str,
        ip_hash: str,
        client_limit: int,
        ip_limit: int,
        concurrency_limit: int,
    ) -> AdmissionResult: ...

    def claim_upload(
        self,
        *,
        user_id: str,
        client_key: str,
        ip_hash: str,
        client_limit: int,
        ip_limit: int,
        concurrency_limit: int,
    ) -> AdmissionResult: ...

    def complete_upload(
        self,
        *,
        client_key: str,
        ip_hash: str,
    ) -> None: ...

    def release_agent_run(self, *, user_id: str) -> None: ...

    def release_ingestion(self, *, user_id: str) -> None: ...

    def reset_client_bucket(self, *, client_key: str) -> None: ...

    def reset_ip_bucket(self, *, ip_hash: str) -> None: ...


class InMemoryQuotaRepository:
    """In-memory quota snapshot store."""

    def __init__(self) -> None:
        self._snapshots: dict[str, QuotaSnapshot] = {}
        self._client_snapshots: dict[str, BucketQuotaSnapshot] = {}
        self._ip_snapshots: dict[str, BucketQuotaSnapshot] = {}
        self._lock = Lock()

    def get_snapshot(self, *, user_id: str) -> QuotaSnapshot:
        return self._snapshots.get(user_id, QuotaSnapshot())

    def get_client_snapshot(
        self,
        *,
        client_key: str,
    ) -> BucketQuotaSnapshot:
        return self._client_snapshots.get(client_key, BucketQuotaSnapshot())

    def get_ip_snapshot(
        self,
        *,
        ip_hash: str,
    ) -> BucketQuotaSnapshot:
        return self._ip_snapshots.get(ip_hash, BucketQuotaSnapshot())

    def set_snapshot(self, *, user_id: str, snapshot: QuotaSnapshot) -> None:
        with self._lock:
            self._snapshots[user_id] = snapshot

    def set_client_snapshot(
        self,
        *,
        client_key: str,
        snapshot: BucketQuotaSnapshot,
    ) -> None:
        with self._lock:
            self._client_snapshots[client_key] = snapshot

    def set_ip_snapshot(
        self,
        *,
        ip_hash: str,
        snapshot: BucketQuotaSnapshot,
    ) -> None:
        with self._lock:
            self._ip_snapshots[ip_hash] = snapshot

    def claim_question(
        self,
        *,
        user_id: str,
        client_key: str,
        ip_hash: str,
        client_limit: int,
        ip_limit: int,
        concurrency_limit: int,
    ) -> AdmissionResult:
        with self._lock:
            user_snapshot = self._snapshots.get(user_id, QuotaSnapshot())
            client_snapshot = self._client_snapshots.get(
                client_key,
                BucketQuotaSnapshot(),
            )
            ip_snapshot = self._ip_snapshots.get(ip_hash, BucketQuotaSnapshot())

            if client_snapshot.questions_used >= client_limit:
                return AdmissionResult(
                    allowed=False,
                    code="browser_question_quota_reached",
                    message="Question quota reached for this browser session.",
                )
            if ip_snapshot.questions_used >= ip_limit:
                return AdmissionResult(
                    allowed=False,
                    code="network_question_quota_reached",
                    message="Question quota reached for this network.",
                )
            if user_snapshot.active_agent_runs >= concurrency_limit:
                return AdmissionResult(
                    allowed=False,
                    code="agent_run_in_progress",
                    message="Another agent run is already in progress.",
                )

            self._snapshots[user_id] = QuotaSnapshot(
                active_ingestions=user_snapshot.active_ingestions,
                active_agent_runs=user_snapshot.active_agent_runs + 1,
            )
            self._client_snapshots[client_key] = BucketQuotaSnapshot(
                questions_used=client_snapshot.questions_used + 1,
                uploads_used=client_snapshot.uploads_used,
            )
            self._ip_snapshots[ip_hash] = BucketQuotaSnapshot(
                questions_used=ip_snapshot.questions_used + 1,
                uploads_used=ip_snapshot.uploads_used,
            )
            return AdmissionResult(allowed=True)

    def claim_upload(
        self,
        *,
        user_id: str,
        client_key: str,
        ip_hash: str,
        client_limit: int,
        ip_limit: int,
        concurrency_limit: int,
    ) -> AdmissionResult:
        with self._lock:
            user_snapshot = self._snapshots.get(user_id, QuotaSnapshot())
            client_snapshot = self._client_snapshots.get(
                client_key,
                BucketQuotaSnapshot(),
            )
            ip_snapshot = self._ip_snapshots.get(ip_hash, BucketQuotaSnapshot())

            if client_snapshot.uploads_used >= client_limit:
                return AdmissionResult(
                    allowed=False,
                    code="browser_upload_quota_reached",
                    message="Upload quota reached for this browser session.",
                )
            if ip_snapshot.uploads_used >= ip_limit:
                return AdmissionResult(
                    allowed=False,
                    code="network_upload_quota_reached",
                    message="Upload quota reached for this network.",
                )
            if user_snapshot.active_ingestions >= concurrency_limit:
                return AdmissionResult(
                    allowed=False,
                    code="ingestion_in_progress",
                    message="Another document is still processing.",
                )

            self._snapshots[user_id] = QuotaSnapshot(
                active_ingestions=user_snapshot.active_ingestions + 1,
                active_agent_runs=user_snapshot.active_agent_runs,
            )
            return AdmissionResult(allowed=True)

    def complete_upload(
        self,
        *,
        client_key: str,
        ip_hash: str,
    ) -> None:
        with self._lock:
            client_snapshot = self._client_snapshots.get(
                client_key,
                BucketQuotaSnapshot(),
            )
            ip_snapshot = self._ip_snapshots.get(ip_hash, BucketQuotaSnapshot())
            self._client_snapshots[client_key] = BucketQuotaSnapshot(
                questions_used=client_snapshot.questions_used,
                uploads_used=client_snapshot.uploads_used + 1,
            )
            self._ip_snapshots[ip_hash] = BucketQuotaSnapshot(
                questions_used=ip_snapshot.questions_used,
                uploads_used=ip_snapshot.uploads_used + 1,
            )

    def release_agent_run(self, *, user_id: str) -> None:
        with self._lock:
            snapshot = self._snapshots.get(user_id, QuotaSnapshot())
            self._snapshots[user_id] = QuotaSnapshot(
                active_ingestions=snapshot.active_ingestions,
                active_agent_runs=max(snapshot.active_agent_runs - 1, 0),
            )

    def release_ingestion(self, *, user_id: str) -> None:
        with self._lock:
            snapshot = self._snapshots.get(user_id, QuotaSnapshot())
            self._snapshots[user_id] = QuotaSnapshot(
                active_ingestions=max(snapshot.active_ingestions - 1, 0),
                active_agent_runs=snapshot.active_agent_runs,
            )

    def reset_client_bucket(self, *, client_key: str) -> None:
        with self._lock:
            self._client_snapshots[client_key] = BucketQuotaSnapshot()

    def reset_ip_bucket(self, *, ip_hash: str) -> None:
        with self._lock:
            self._ip_snapshots[ip_hash] = BucketQuotaSnapshot()


class QuotaService:
    """Builds quota summaries and admission snapshots."""

    def __init__(
        self,
        policy: QuotaPolicy,
        repository: QuotaRepository,
        client_key: str | None = None,
        ip_hash: str | None = None,
    ) -> None:
        self.policy = policy
        self.repository = repository
        self.client_key = client_key
        self.ip_hash = ip_hash

    def claim_question(
        self,
        *,
        user_id: str,
        client_key: str | None = None,
        ip_hash: str | None = None,
    ) -> AdmissionResult:
        return self.repository.claim_question(
            user_id=user_id,
            client_key=self._resolve_client_key(
                user_id=user_id,
                client_key=client_key,
            ),
            ip_hash=self._resolve_ip_hash(ip_hash=ip_hash),
            client_limit=self.policy.user_question_limit,
            ip_limit=self.policy.ip_question_limit,
            concurrency_limit=self.policy.concurrent_agent_runs_per_user,
        )

    def claim_upload(
        self,
        *,
        user_id: str,
        client_key: str | None = None,
        ip_hash: str | None = None,
    ) -> AdmissionResult:
        return self.repository.claim_upload(
            user_id=user_id,
            client_key=self._resolve_client_key(
                user_id=user_id,
                client_key=client_key,
            ),
            ip_hash=self._resolve_ip_hash(ip_hash=ip_hash),
            client_limit=self.policy.user_upload_limit,
            ip_limit=self.policy.ip_upload_limit,
            concurrency_limit=self.policy.concurrent_ingestions_per_user,
        )

    def release_agent_run(self, *, user_id: str) -> None:
        self.repository.release_agent_run(user_id=user_id)

    def release_ingestion(self, *, user_id: str) -> None:
        self.repository.release_ingestion(user_id=user_id)

    def complete_upload(
        self,
        *,
        client_key: str | None = None,
        ip_hash: str | None = None,
        user_id: str | None = None,
    ) -> None:
        resolved_user_id = user_id or "unknown-user"
        self.repository.complete_upload(
            client_key=self._resolve_client_key(
                user_id=resolved_user_id,
                client_key=client_key,
            ),
            ip_hash=self._resolve_ip_hash(ip_hash=ip_hash),
        )

    def get_session_quota_summary(
        self,
        *,
        user_id: str,
        token_usage: TokenUsageSummary,
    ) -> SessionQuotaSummary:
        client_snapshot = self.repository.get_client_snapshot(
            client_key=self._resolve_client_key(user_id=user_id, client_key=None)
        )
        return SessionQuotaSummary(
            questions=QuotaBucket(
                limit=self.policy.user_question_limit,
                used=client_snapshot.questions_used,
            ),
            uploads=QuotaBucket(
                limit=self.policy.user_upload_limit,
                used=client_snapshot.uploads_used,
            ),
            tokens=token_usage,
        )

    def _resolve_client_key(self, *, user_id: str, client_key: str | None) -> str:
        return client_key or self.client_key or user_id

    def _resolve_ip_hash(self, *, ip_hash: str | None) -> str:
        return ip_hash or self.ip_hash or "unknown-network"
