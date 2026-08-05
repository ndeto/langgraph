from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Engine,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    and_,
    create_engine,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert

from atlasai.application.documents import (
    DocumentRecord,
    IngestionEventRecord,
    IngestionJobRecord,
    InMemoryDocumentRepository,
)
from atlasai.application.quotas import (
    AdmissionResult,
    BucketQuotaSnapshot,
    InMemoryQuotaRepository,
    QuotaSnapshot,
)
from atlasai.application.sessions import CleanupJobRecord, InMemorySessionRepository
from atlasai.application.threads import (
    InMemoryThreadRepository,
    ThreadMessageRecord,
    ThreadRecord,
)
from atlasai.application.usage import InMemoryUsageRepository, UsageRecord
from atlasai.config.sys_config import get_env
from atlasai.domain.models import AnonymousSession

metadata = MetaData()

sessions_table = Table(
    "demo_sessions",
    metadata,
    Column("user_id", String, primary_key=True),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("active_document", String, nullable=True),
    Column("active_thread", String, nullable=True),
)

documents_table = Table(
    "documents",
    metadata,
    Column("document_id", String, primary_key=True),
    Column("user_id", String, nullable=False, index=True),
    Column("filename", String, nullable=False),
    Column("size_bytes", BigInteger, nullable=False),
    Column("status", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("is_active", Boolean, nullable=False, default=False),
)

ingestion_jobs_table = Table(
    "ingestion_jobs",
    metadata,
    Column("job_id", String, primary_key=True),
    Column("user_id", String, nullable=False, index=True),
    Column("document_id", String, nullable=False, index=True),
    Column("state", String, nullable=False),
    Column("source_path", String, nullable=True),
    Column("attempts", Integer, nullable=False, default=0),
    Column("heartbeat_at", DateTime(timezone=True), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("failure_text", String, nullable=True),
)

ingestion_events_table = Table(
    "ingestion_events",
    metadata,
    Column("job_id", String, nullable=False, index=True),
    Column("event_id", Integer, nullable=False),
    Column("payload", JSON, nullable=False),
    UniqueConstraint("job_id", "event_id", name="uq_ingestion_event_job_id"),
)

cleanup_jobs_table = Table(
    "cleanup_jobs",
    metadata,
    Column("job_id", String, primary_key=True),
    Column("user_id", String, nullable=False, index=True),
    Column("state", String, nullable=False),
    Column("attempts", Integer, nullable=False, default=0),
    Column("heartbeat_at", DateTime(timezone=True), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("failure_text", String, nullable=True),
)

threads_table = Table(
    "threads",
    metadata,
    Column("thread_id", String, primary_key=True),
    Column("user_id", String, nullable=False, index=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column("document_id", String, nullable=True),
)

user_quotas_table = Table(
    "user_quotas",
    metadata,
    Column("user_id", String, primary_key=True),
    Column("questions_used", Integer, nullable=False, default=0),
    Column("uploads_used", Integer, nullable=False, default=0),
    Column("active_ingestions", Integer, nullable=False, default=0),
    Column("active_agent_runs", Integer, nullable=False, default=0),
)

request_key_quotas_table = Table(
    "request_key_quotas",
    metadata,
    Column("request_key", String, primary_key=True),
    Column("questions_used", Integer, nullable=False, default=0),
    Column("uploads_used", Integer, nullable=False, default=0),
)

client_key_quotas_table = Table(
    "client_key_quotas",
    metadata,
    Column("client_key", String, primary_key=True),
    Column("questions_used", Integer, nullable=False, default=0),
    Column("uploads_used", Integer, nullable=False, default=0),
)

thread_messages_table = Table(
    "thread_messages",
    metadata,
    Column("message_id", String, primary_key=True),
    Column("thread_id", String, nullable=False, index=True),
    Column("role", String, nullable=False),
    Column("content", String, nullable=False),
    Column("status", String, nullable=False),
    Column("assets", JSON, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

usage_records_table = Table(
    "usage_records",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String, nullable=False, index=True),
    Column("operation", String, nullable=False),
    Column("run_id", String, nullable=True),
    Column("provider", String, nullable=True),
    Column("model", String, nullable=True),
    Column("input_tokens", Integer, nullable=True),
    Column("output_tokens", Integer, nullable=True),
    Column("total_tokens", Integer, nullable=True),
    Column("status", String, nullable=False),
    Column("trace_id", String, nullable=True),
    UniqueConstraint("user_id", "run_id", name="uq_usage_user_run"),
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _to_document_record(row: Any) -> DocumentRecord:
    return DocumentRecord(
        document_id=row.document_id,
        user_id=row.user_id,
        filename=row.filename,
        size_bytes=row.size_bytes,
        status=row.status,
        created_at=row.created_at,
        expires_at=row.expires_at,
    )


def _to_thread_record(row: Any) -> ThreadRecord:
    return ThreadRecord(
        thread_id=row.thread_id,
        user_id=row.user_id,
        created_at=row.created_at,
        expires_at=row.expires_at,
        document_id=row.document_id,
    )


def _to_usage_record(row: Any) -> UsageRecord:
    return UsageRecord(
        operation=row.operation,
        run_id=row.run_id,
        provider=row.provider,
        model=row.model,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        total_tokens=row.total_tokens,
        status=row.status,
        trace_id=row.trace_id,
    )


def _to_thread_message_record(row: Any) -> ThreadMessageRecord:
    assets = row.assets if isinstance(row.assets, list) else None
    return ThreadMessageRecord(
        message_id=row.message_id,
        thread_id=row.thread_id,
        role=row.role,
        content=row.content,
        created_at=row.created_at,
        assets=assets,
        status=row.status,
    )


def _to_cleanup_job_record(row: Any) -> CleanupJobRecord:
    return CleanupJobRecord(
        job_id=row.job_id,
        user_id=row.user_id,
        state=row.state,
        created_at=row.created_at,
        updated_at=row.updated_at,
        attempts=row.attempts,
        heartbeat_at=row.heartbeat_at,
        failure_text=row.failure_text,
    )


class PostgresSessionRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def get_session(self, *, user_id: str) -> AnonymousSession | None:
        with self.engine.begin() as conn:
            row = (
                conn.execute(
                    select(sessions_table).where(
                        and_(
                            sessions_table.c.user_id == user_id,
                            sessions_table.c.expires_at > _utcnow(),
                        )
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return AnonymousSession(
            user_id=row["user_id"],
            expires_at=row["expires_at"],
            active_document=row["active_document"],
            active_thread=row["active_thread"],
        )

    def upsert_session(self, session: AnonymousSession) -> None:
        statement = pg_insert(sessions_table).values(
            user_id=session.user_id,
            expires_at=session.expires_at,
            active_document=session.active_document,
            active_thread=session.active_thread,
        )
        with self.engine.begin() as conn:
            conn.execute(
                statement.on_conflict_do_update(
                    index_elements=[sessions_table.c.user_id],
                    set_={
                        "expires_at": session.expires_at,
                        "active_document": session.active_document,
                        "active_thread": session.active_thread,
                    },
                )
            )

    def expire_session(self, *, user_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(sessions_table)
                .where(sessions_table.c.user_id == user_id)
                .values(
                    expires_at=_utcnow(),
                    active_document=None,
                    active_thread=None,
                )
            )

    def enqueue_cleanup_job(self, *, user_id: str) -> CleanupJobRecord:
        now = _utcnow()
        job = CleanupJobRecord(
            job_id=str(uuid4()),
            user_id=user_id,
            state="queued",
            created_at=now,
            updated_at=now,
        )
        with self.engine.begin() as conn:
            conn.execute(
                insert(cleanup_jobs_table).values(
                    job_id=job.job_id,
                    user_id=job.user_id,
                    state=job.state,
                    attempts=job.attempts,
                    heartbeat_at=job.heartbeat_at,
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                    failure_text=job.failure_text,
                )
            )
        return job

    def claim_next_cleanup_job(
        self,
        *,
        heartbeat_timeout_seconds: int,
        max_attempts: int,
    ) -> CleanupJobRecord | None:
        stale_before = _utcnow() - timedelta(seconds=heartbeat_timeout_seconds)
        with self.engine.begin() as conn:
            row = (
                conn.execute(
                    select(cleanup_jobs_table)
                    .where(
                        (
                            cleanup_jobs_table.c.state == "queued"
                        )
                        | (
                            and_(
                                cleanup_jobs_table.c.state == "processing",
                                cleanup_jobs_table.c.heartbeat_at.is_not(None),
                                cleanup_jobs_table.c.heartbeat_at < stale_before,
                                cleanup_jobs_table.c.attempts < max_attempts,
                            )
                        )
                    )
                    .order_by(cleanup_jobs_table.c.created_at.asc())
                    .with_for_update(skip_locked=True)
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if row is None:
                return None

            now = _utcnow()
            conn.execute(
                update(cleanup_jobs_table)
                .where(cleanup_jobs_table.c.job_id == row["job_id"])
                .values(
                    state="processing",
                    attempts=row["attempts"] + 1,
                    heartbeat_at=now,
                    updated_at=now,
                    failure_text=None,
                )
            )
            claimed = (
                conn.execute(
                    select(cleanup_jobs_table).where(
                        cleanup_jobs_table.c.job_id == row["job_id"]
                    )
                )
                .mappings()
                .one()
            )
        return _to_cleanup_job_record(claimed)

    def heartbeat_cleanup_job(self, *, job_id: str) -> None:
        now = _utcnow()
        with self.engine.begin() as conn:
            conn.execute(
                update(cleanup_jobs_table)
                .where(cleanup_jobs_table.c.job_id == job_id)
                .values(heartbeat_at=now, updated_at=now)
            )

    def mark_cleanup_job_ready(self, *, job_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(cleanup_jobs_table)
                .where(cleanup_jobs_table.c.job_id == job_id)
                .values(state="ready", updated_at=_utcnow(), failure_text=None)
            )

    def mark_cleanup_job_failed(self, *, job_id: str, failure_text: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(cleanup_jobs_table)
                .where(cleanup_jobs_table.c.job_id == job_id)
                .values(
                    state="failed",
                    updated_at=_utcnow(),
                    failure_text=failure_text,
                )
            )


class PostgresThreadRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def create_thread(
        self,
        *,
        user_id: str,
        expires_at: datetime,
        document_id: str | None = None,
    ) -> ThreadRecord:
        thread = ThreadRecord(
            thread_id=str(uuid4()),
            user_id=user_id,
            created_at=datetime.now(expires_at.tzinfo),
            expires_at=expires_at,
            document_id=document_id,
        )
        with self.engine.begin() as conn:
            conn.execute(
                insert(threads_table).values(
                    thread_id=thread.thread_id,
                    user_id=thread.user_id,
                    created_at=thread.created_at,
                    expires_at=thread.expires_at,
                    document_id=thread.document_id,
                )
            )
        return thread

    def get_thread(self, *, thread_id: str) -> ThreadRecord | None:
        with self.engine.begin() as conn:
            row = (
                conn.execute(
                    select(threads_table).where(threads_table.c.thread_id == thread_id)
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return _to_thread_record(row)

    def append_message(
        self,
        *,
        thread_id: str,
        role: str,
        content: str,
        assets: list[dict[str, str]] | None = None,
        status: str = "done",
    ) -> ThreadMessageRecord:
        message = ThreadMessageRecord(
            message_id=str(uuid4()),
            thread_id=thread_id,
            role=role,
            content=content,
            created_at=_utcnow(),
            assets=assets,
            status=status,
        )
        with self.engine.begin() as conn:
            conn.execute(
                insert(thread_messages_table).values(
                    message_id=message.message_id,
                    thread_id=message.thread_id,
                    role=message.role,
                    content=message.content,
                    status=message.status,
                    assets=message.assets,
                    created_at=message.created_at,
                )
            )
        return message

    def list_messages(self, *, thread_id: str) -> list[ThreadMessageRecord]:
        with self.engine.begin() as conn:
            rows = (
                conn.execute(
                    select(thread_messages_table)
                    .where(thread_messages_table.c.thread_id == thread_id)
                    .order_by(thread_messages_table.c.created_at.asc())
                )
                .mappings()
                .all()
            )
        return [_to_thread_message_record(row) for row in rows]


class PostgresUsageRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def list_records(self, *, user_id: str) -> list[UsageRecord]:
        with self.engine.begin() as conn:
            rows = (
                conn.execute(
                    select(usage_records_table)
                    .where(usage_records_table.c.user_id == user_id)
                    .order_by(usage_records_table.c.id.asc())
                )
                .mappings()
                .all()
            )
        return [_to_usage_record(row) for row in rows]

    def append_record(self, *, user_id: str, record: UsageRecord) -> bool:
        statement = pg_insert(usage_records_table).values(
            user_id=user_id,
            operation=record.operation,
            run_id=record.run_id,
            provider=record.provider,
            model=record.model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            total_tokens=record.total_tokens,
            status=record.status,
            trace_id=record.trace_id,
        )
        with self.engine.begin() as conn:
            if record.run_id is None:
                conn.execute(statement)
                return True

            result = conn.execute(
                statement.on_conflict_do_nothing(constraint="uq_usage_user_run")
            )
            return result.rowcount > 0


class PostgresQuotaRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def get_snapshot(self, *, user_id: str) -> QuotaSnapshot:
        with self.engine.begin() as conn:
            row = self._get_or_create_user_quota(conn, user_id=user_id)
        return QuotaSnapshot(
            active_ingestions=row["active_ingestions"],
            active_agent_runs=row["active_agent_runs"],
        )

    def get_client_snapshot(
        self,
        *,
        client_key: str,
    ) -> BucketQuotaSnapshot:
        with self.engine.begin() as conn:
            row = self._get_or_create_client_key_quota(conn, client_key=client_key)
        return BucketQuotaSnapshot(
            questions_used=row["questions_used"],
            uploads_used=row["uploads_used"],
        )

    def get_ip_snapshot(
        self,
        *,
        ip_hash: str,
    ) -> BucketQuotaSnapshot:
        with self.engine.begin() as conn:
            row = self._get_or_create_request_key_quota(conn, request_key=ip_hash)
        return BucketQuotaSnapshot(
            questions_used=row["questions_used"],
            uploads_used=row["uploads_used"],
        )

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
        with self.engine.begin() as conn:
            user_row = self._get_or_create_user_quota(conn, user_id=user_id, lock=True)
            client_key_row = self._get_or_create_client_key_quota(
                conn,
                client_key=client_key,
                lock=True,
            )
            ip_row = self._get_or_create_request_key_quota(
                conn,
                request_key=ip_hash,
                lock=True,
            )

            if client_key_row["questions_used"] >= client_limit:
                return AdmissionResult(
                    False,
                    "browser_question_quota_reached",
                    "Question quota reached for this browser session.",
                )
            if ip_row["questions_used"] >= ip_limit:
                return AdmissionResult(
                    False,
                    "network_question_quota_reached",
                    "Question quota reached for this network.",
                )
            if user_row["active_agent_runs"] >= concurrency_limit:
                return AdmissionResult(
                    False,
                    "agent_run_in_progress",
                    "Another agent run is already in progress.",
                )

            conn.execute(
                update(user_quotas_table)
                .where(user_quotas_table.c.user_id == user_id)
                .values(
                    active_agent_runs=user_row["active_agent_runs"] + 1,
                )
            )
            conn.execute(
                update(client_key_quotas_table)
                .where(client_key_quotas_table.c.client_key == client_key)
                .values(questions_used=client_key_row["questions_used"] + 1)
            )
            conn.execute(
                update(request_key_quotas_table)
                .where(request_key_quotas_table.c.request_key == ip_hash)
                .values(questions_used=ip_row["questions_used"] + 1)
            )
        return AdmissionResult(True, None, None)

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
        with self.engine.begin() as conn:
            user_row = self._get_or_create_user_quota(conn, user_id=user_id, lock=True)
            client_key_row = self._get_or_create_client_key_quota(
                conn,
                client_key=client_key,
                lock=True,
            )
            ip_row = self._get_or_create_request_key_quota(
                conn,
                request_key=ip_hash,
                lock=True,
            )

            if client_key_row["uploads_used"] >= client_limit:
                return AdmissionResult(
                    False,
                    "browser_upload_quota_reached",
                    "Upload quota reached for this browser session.",
                )
            if ip_row["uploads_used"] >= ip_limit:
                return AdmissionResult(
                    False,
                    "network_upload_quota_reached",
                    "Upload quota reached for this network.",
                )
            if user_row["active_ingestions"] >= concurrency_limit:
                return AdmissionResult(
                    False,
                    "ingestion_in_progress",
                    "Another document is still processing.",
                )

            conn.execute(
                update(user_quotas_table)
                .where(user_quotas_table.c.user_id == user_id)
                .values(
                    active_ingestions=user_row["active_ingestions"] + 1,
                )
            )
            conn.execute(
                update(client_key_quotas_table)
                .where(client_key_quotas_table.c.client_key == client_key)
                .values(uploads_used=client_key_row["uploads_used"] + 1)
            )
            conn.execute(
                update(request_key_quotas_table)
                .where(request_key_quotas_table.c.request_key == ip_hash)
                .values(uploads_used=ip_row["uploads_used"] + 1)
            )
        return AdmissionResult(True, None, None)

    def release_agent_run(self, *, user_id: str) -> None:
        with self.engine.begin() as conn:
            row = self._get_or_create_user_quota(conn, user_id=user_id, lock=True)
            conn.execute(
                update(user_quotas_table)
                .where(user_quotas_table.c.user_id == user_id)
                .values(active_agent_runs=max(row["active_agent_runs"] - 1, 0))
            )

    def release_ingestion(self, *, user_id: str) -> None:
        with self.engine.begin() as conn:
            row = self._get_or_create_user_quota(conn, user_id=user_id, lock=True)
            conn.execute(
                update(user_quotas_table)
                .where(user_quotas_table.c.user_id == user_id)
                .values(active_ingestions=max(row["active_ingestions"] - 1, 0))
            )

    def reset_client_bucket(self, *, client_key: str) -> None:
        with self.engine.begin() as conn:
            self._get_or_create_client_key_quota(conn, client_key=client_key, lock=True)
            conn.execute(
                update(client_key_quotas_table)
                .where(client_key_quotas_table.c.client_key == client_key)
                .values(questions_used=0, uploads_used=0)
            )

    def reset_ip_bucket(self, *, ip_hash: str) -> None:
        with self.engine.begin() as conn:
            self._get_or_create_request_key_quota(
                conn,
                request_key=ip_hash,
                lock=True,
            )
            conn.execute(
                update(request_key_quotas_table)
                .where(request_key_quotas_table.c.request_key == ip_hash)
                .values(questions_used=0, uploads_used=0)
            )

    @staticmethod
    def _get_or_create_user_quota(conn, *, user_id: str, lock: bool = False):
        statement = select(user_quotas_table).where(
            user_quotas_table.c.user_id == user_id
        )
        if lock:
            statement = statement.with_for_update()
        row = conn.execute(statement).mappings().first()
        if row is not None:
            return row
        conn.execute(insert(user_quotas_table).values(user_id=user_id))
        return conn.execute(statement).mappings().one()

    @staticmethod
    def _get_or_create_request_key_quota(conn, *, request_key: str, lock: bool = False):
        statement = select(request_key_quotas_table).where(
            request_key_quotas_table.c.request_key == request_key
        )
        if lock:
            statement = statement.with_for_update()
        row = conn.execute(statement).mappings().first()
        if row is not None:
            return row
        conn.execute(insert(request_key_quotas_table).values(request_key=request_key))
        return conn.execute(statement).mappings().one()

    @staticmethod
    def _get_or_create_client_key_quota(conn, *, client_key: str, lock: bool = False):
        statement = select(client_key_quotas_table).where(
            client_key_quotas_table.c.client_key == client_key
        )
        if lock:
            statement = statement.with_for_update()
        row = conn.execute(statement).mappings().first()
        if row is not None:
            return row
        conn.execute(insert(client_key_quotas_table).values(client_key=client_key))
        return conn.execute(statement).mappings().one()


class PostgresDocumentRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def create_job(
        self,
        *,
        user_id: str,
        filename: str,
        size_bytes: int,
        ttl_seconds: int,
        source_path: str,
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        created_at = _utcnow()
        expires_at = created_at + timedelta(seconds=ttl_seconds)
        document = DocumentRecord(
            document_id=str(uuid4()),
            user_id=user_id,
            filename=filename,
            size_bytes=size_bytes,
            status="queued",
            created_at=created_at,
            expires_at=expires_at,
        )
        job = IngestionJobRecord(
            job_id=str(uuid4()),
            user_id=user_id,
            document_id=document.document_id,
            state="queued",
            created_at=created_at,
            expires_at=expires_at,
            source_path=source_path,
            updated_at=created_at,
        )
        with self.engine.begin() as conn:
            conn.execute(
                insert(documents_table).values(
                    document_id=document.document_id,
                    user_id=document.user_id,
                    filename=document.filename,
                    size_bytes=document.size_bytes,
                    status=document.status,
                    created_at=document.created_at,
                    expires_at=document.expires_at,
                    is_active=False,
                )
            )
            conn.execute(
                insert(ingestion_jobs_table).values(
                    job_id=job.job_id,
                    user_id=job.user_id,
                    document_id=job.document_id,
                    state=job.state,
                    source_path=job.source_path,
                    attempts=job.attempts,
                    heartbeat_at=job.heartbeat_at,
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                    expires_at=job.expires_at,
                    failure_text=job.failure_text,
                )
            )
            self._append_event(
                conn,
                job_id=job.job_id,
                payload={"type": "queued", "text": "Document queued"},
            )
        return document, job

    def get_document(self, *, document_id: str) -> DocumentRecord | None:
        with self.engine.begin() as conn:
            row = (
                conn.execute(
                    select(documents_table).where(
                        documents_table.c.document_id == document_id
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return _to_document_record(row)

    def get_job(self, *, job_id: str) -> IngestionJobRecord | None:
        with self.engine.begin() as conn:
            job_row = (
                conn.execute(
                    select(ingestion_jobs_table).where(
                        ingestion_jobs_table.c.job_id == job_id
                    )
                )
                .mappings()
                .first()
            )
            if job_row is None:
                return None
            event_rows = (
                conn.execute(
                    select(ingestion_events_table)
                    .where(ingestion_events_table.c.job_id == job_id)
                    .order_by(ingestion_events_table.c.event_id.asc())
                )
                .mappings()
                .all()
            )
        return IngestionJobRecord(
            job_id=job_row["job_id"],
            user_id=job_row["user_id"],
            document_id=job_row["document_id"],
            state=job_row["state"],
            created_at=job_row["created_at"],
            expires_at=job_row["expires_at"],
            source_path=job_row["source_path"],
            attempts=job_row["attempts"],
            heartbeat_at=job_row["heartbeat_at"],
            updated_at=job_row["updated_at"],
            failure_text=job_row["failure_text"],
            events=[
                IngestionEventRecord(event_id=row["event_id"], payload=row["payload"])
                for row in event_rows
            ],
        )

    def get_active_document(self, *, user_id: str) -> DocumentRecord | None:
        with self.engine.begin() as conn:
            row = (
                conn.execute(
                    select(documents_table).where(
                        and_(
                            documents_table.c.user_id == user_id,
                            documents_table.c.is_active.is_(True),
                        )
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            return None
        return _to_document_record(row)

    def list_documents(self, *, user_id: str) -> list[DocumentRecord]:
        with self.engine.begin() as conn:
            rows = (
                conn.execute(
                    select(documents_table)
                    .where(documents_table.c.user_id == user_id)
                    .order_by(documents_table.c.created_at.asc())
                )
                .mappings()
                .all()
            )
        return [_to_document_record(row) for row in rows]

    def list_user_source_paths(self, *, user_id: str) -> list[str]:
        with self.engine.begin() as conn:
            rows = (
                conn.execute(
                    select(ingestion_jobs_table.c.source_path).where(
                        and_(
                            ingestion_jobs_table.c.user_id == user_id,
                            ingestion_jobs_table.c.source_path.is_not(None),
                        )
                    )
                )
                .scalars()
                .all()
            )
        return [row for row in rows if isinstance(row, str)]

    def append_event(self, *, job_id: str, payload: dict) -> None:
        with self.engine.begin() as conn:
            self._append_event(conn, job_id=job_id, payload=payload)

    def claim_next_job(
        self,
        *,
        heartbeat_timeout_seconds: int,
        max_attempts: int,
    ) -> IngestionJobRecord | None:
        stale_before = _utcnow() - timedelta(seconds=heartbeat_timeout_seconds)
        with self.engine.begin() as conn:
            row = (
                conn.execute(
                    select(ingestion_jobs_table)
                    .where(
                        (ingestion_jobs_table.c.state == "queued")
                        | (
                            and_(
                                ingestion_jobs_table.c.state == "processing",
                                ingestion_jobs_table.c.heartbeat_at.is_not(None),
                                ingestion_jobs_table.c.heartbeat_at < stale_before,
                                ingestion_jobs_table.c.attempts < max_attempts,
                            )
                        )
                    )
                    .where(ingestion_jobs_table.c.expires_at > _utcnow())
                    .order_by(ingestion_jobs_table.c.created_at.asc())
                    .with_for_update(skip_locked=True)
                    .limit(1)
                )
                .mappings()
                .first()
            )
            if row is None:
                return None

            now = _utcnow()
            conn.execute(
                update(ingestion_jobs_table)
                .where(ingestion_jobs_table.c.job_id == row["job_id"])
                .values(
                    state="processing",
                    attempts=row["attempts"] + 1,
                    heartbeat_at=now,
                    updated_at=now,
                    failure_text=None,
                )
            )
        return self.get_job(job_id=row["job_id"])

    def heartbeat_job(self, *, job_id: str) -> None:
        now = _utcnow()
        with self.engine.begin() as conn:
            conn.execute(
                update(ingestion_jobs_table)
                .where(ingestion_jobs_table.c.job_id == job_id)
                .values(heartbeat_at=now, updated_at=now)
            )

    def mark_ready(self, *, job_id: str) -> None:
        with self.engine.begin() as conn:
            job = (
                conn.execute(
                    select(ingestion_jobs_table)
                    .where(ingestion_jobs_table.c.job_id == job_id)
                    .with_for_update()
                )
                .mappings()
                .one()
            )
            conn.execute(
                update(ingestion_jobs_table)
                .where(ingestion_jobs_table.c.job_id == job_id)
                .values(
                    state="ready",
                    updated_at=_utcnow(),
                    failure_text=None,
                )
            )
            conn.execute(
                update(documents_table)
                .where(documents_table.c.user_id == job["user_id"])
                .values(is_active=False)
            )
            conn.execute(
                update(documents_table)
                .where(documents_table.c.document_id == job["document_id"])
                .values(status="ready", is_active=True)
            )
            self._append_event(
                conn, job_id=job_id, payload={"type": "ready", "text": "Document ready"}
            )

    def mark_failed(self, *, job_id: str, text: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(ingestion_jobs_table)
                .where(ingestion_jobs_table.c.job_id == job_id)
                .values(
                    state="failed",
                    updated_at=_utcnow(),
                    failure_text=text,
                )
            )
            job = (
                conn.execute(
                    select(ingestion_jobs_table).where(
                        ingestion_jobs_table.c.job_id == job_id
                    )
                )
                .mappings()
                .one()
            )
            conn.execute(
                update(documents_table)
                .where(documents_table.c.document_id == job["document_id"])
                .values(status="failed")
            )
            self._append_event(
                conn, job_id=job_id, payload={"type": "failed", "text": text}
            )

    @staticmethod
    def _append_event(conn, *, job_id: str, payload: dict) -> None:
        next_event_id = (
            conn.execute(
                select(ingestion_events_table.c.event_id)
                .where(ingestion_events_table.c.job_id == job_id)
                .order_by(ingestion_events_table.c.event_id.desc())
                .limit(1)
            ).scalar_one_or_none()
            or 0
        ) + 1
        conn.execute(
            insert(ingestion_events_table).values(
                job_id=job_id,
                event_id=next_event_id,
                payload=payload,
            )
        )


@dataclass
class InMemoryRepositoryBundle:
    """In-memory repository bundle for tests."""

    sessions: InMemorySessionRepository
    documents: InMemoryDocumentRepository
    quotas: InMemoryQuotaRepository
    threads: InMemoryThreadRepository
    usage: InMemoryUsageRepository

    def startup(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def expire_user_data(self, *, user_id: str) -> None:
        self.sessions.expire_session(user_id=user_id)
        self.documents._jobs = {
            job_id: job
            for job_id, job in self.documents._jobs.items()
            if job.user_id != user_id
        }
        self.documents._documents = {
            document_id: document
            for document_id, document in self.documents._documents.items()
            if document.user_id != user_id
        }
        self.documents._active_document_by_user.pop(user_id, None)
        self.threads._threads = {
            thread_id: thread
            for thread_id, thread in self.threads._threads.items()
            if thread.user_id != user_id
        }
        self.threads._messages_by_thread = {
            thread_id: messages
            for thread_id, messages in self.threads._messages_by_thread.items()
            if thread_id in self.threads._threads
        }
        self.quotas._snapshots.pop(user_id, None)
        self.usage._records.pop(user_id, None)
        self.usage._seen_run_ids.pop(user_id, None)

    def delete_user_data(self, *, user_id: str) -> None:
        self.expire_user_data(user_id=user_id)

    def enqueue_cleanup_job(self, *, user_id: str) -> CleanupJobRecord:
        return self.sessions.enqueue_cleanup_job(user_id=user_id)

    def claim_next_cleanup_job(
        self,
        *,
        heartbeat_timeout_seconds: int,
        max_attempts: int,
    ) -> CleanupJobRecord | None:
        return self.sessions.claim_next_cleanup_job(
            heartbeat_timeout_seconds=heartbeat_timeout_seconds,
            max_attempts=max_attempts,
        )

    def heartbeat_cleanup_job(self, *, job_id: str) -> None:
        self.sessions.heartbeat_cleanup_job(job_id=job_id)

    def mark_cleanup_job_ready(self, *, job_id: str) -> None:
        self.sessions.mark_cleanup_job_ready(job_id=job_id)

    def mark_cleanup_job_failed(self, *, job_id: str, failure_text: str) -> None:
        self.sessions.mark_cleanup_job_failed(
            job_id=job_id,
            failure_text=failure_text,
        )


class PostgresRepositoryBundle:
    """Warmed Postgres repositories."""

    def __init__(self, conn_string: str | None = None) -> None:
        self.conn_string = conn_string or get_env("DB_CONN")
        self.engine: Engine | None = None
        self.sessions: PostgresSessionRepository | None = None
        self.documents: PostgresDocumentRepository | None = None
        self.quotas: PostgresQuotaRepository | None = None
        self.threads: PostgresThreadRepository | None = None
        self.usage: PostgresUsageRepository | None = None

    async def startup(self) -> None:
        if self.engine is not None:
            return
        self.engine = create_engine(self.conn_string, future=True)
        self.sessions = PostgresSessionRepository(self.engine)
        self.documents = PostgresDocumentRepository(self.engine)
        self.quotas = PostgresQuotaRepository(self.engine)
        self.threads = PostgresThreadRepository(self.engine)
        self.usage = PostgresUsageRepository(self.engine)

    async def shutdown(self) -> None:
        if self.engine is not None:
            self.engine.dispose()
        self.engine = None

    def expire_user_data(self, *, user_id: str) -> None:
        if self.engine is None:
            raise RuntimeError("Repository bundle startup must run before use.")

        now = _utcnow()
        with self.engine.begin() as conn:
            conn.execute(
                update(sessions_table)
                .where(sessions_table.c.user_id == user_id)
                .values(expires_at=now, active_document=None, active_thread=None)
            )
            conn.execute(
                update(documents_table)
                .where(documents_table.c.user_id == user_id)
                .values(expires_at=now, is_active=False)
            )
            conn.execute(
                update(ingestion_jobs_table)
                .where(ingestion_jobs_table.c.user_id == user_id)
                .values(expires_at=now)
            )
            conn.execute(
                update(threads_table)
                .where(threads_table.c.user_id == user_id)
                .values(expires_at=now)
            )
            conn.execute(
                update(user_quotas_table)
                .where(user_quotas_table.c.user_id == user_id)
                .values(active_ingestions=0, active_agent_runs=0)
            )

    def delete_user_data(self, *, user_id: str) -> None:
        if self.engine is None:
            raise RuntimeError("Repository bundle startup must run before use.")

        with self.engine.begin() as conn:
            job_ids = list(
                conn.execute(
                    select(ingestion_jobs_table.c.job_id).where(
                        ingestion_jobs_table.c.user_id == user_id
                    )
                ).scalars()
            )
            if job_ids:
                conn.execute(
                    delete(ingestion_events_table).where(
                        ingestion_events_table.c.job_id.in_(job_ids)
                    )
                )
            conn.execute(
                delete(ingestion_jobs_table).where(
                    ingestion_jobs_table.c.user_id == user_id
                )
            )
            thread_ids = list(
                conn.execute(
                    select(threads_table.c.thread_id).where(threads_table.c.user_id == user_id)
                ).scalars()
            )
            if thread_ids:
                conn.execute(
                    delete(thread_messages_table).where(
                        thread_messages_table.c.thread_id.in_(thread_ids)
                    )
                )
            conn.execute(delete(documents_table).where(documents_table.c.user_id == user_id))
            conn.execute(delete(threads_table).where(threads_table.c.user_id == user_id))
            conn.execute(
                delete(usage_records_table).where(usage_records_table.c.user_id == user_id)
            )
            conn.execute(delete(user_quotas_table).where(user_quotas_table.c.user_id == user_id))
            conn.execute(delete(sessions_table).where(sessions_table.c.user_id == user_id))

    def enqueue_cleanup_job(self, *, user_id: str) -> CleanupJobRecord:
        if self.sessions is None:
            raise RuntimeError("Repository bundle startup must run before use.")
        return self.sessions.enqueue_cleanup_job(user_id=user_id)

    def claim_next_cleanup_job(
        self,
        *,
        heartbeat_timeout_seconds: int,
        max_attempts: int,
    ) -> CleanupJobRecord | None:
        if self.sessions is None:
            raise RuntimeError("Repository bundle startup must run before use.")
        return self.sessions.claim_next_cleanup_job(
            heartbeat_timeout_seconds=heartbeat_timeout_seconds,
            max_attempts=max_attempts,
        )

    def heartbeat_cleanup_job(self, *, job_id: str) -> None:
        if self.sessions is None:
            raise RuntimeError("Repository bundle startup must run before use.")
        self.sessions.heartbeat_cleanup_job(job_id=job_id)

    def mark_cleanup_job_ready(self, *, job_id: str) -> None:
        if self.sessions is None:
            raise RuntimeError("Repository bundle startup must run before use.")
        self.sessions.mark_cleanup_job_ready(job_id=job_id)

    def mark_cleanup_job_failed(self, *, job_id: str, failure_text: str) -> None:
        if self.sessions is None:
            raise RuntimeError("Repository bundle startup must run before use.")
        self.sessions.mark_cleanup_job_failed(
            job_id=job_id,
            failure_text=failure_text,
        )


def build_in_memory_repository_bundle() -> InMemoryRepositoryBundle:
    return InMemoryRepositoryBundle(
        sessions=InMemorySessionRepository(),
        documents=InMemoryDocumentRepository(),
        quotas=InMemoryQuotaRepository(),
        threads=InMemoryThreadRepository(),
        usage=InMemoryUsageRepository(),
    )
