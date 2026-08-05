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
    InMemoryQuotaRepository,
    QuotaSnapshot,
    RequestKeyQuotaSnapshot,
)
from atlasai.application.sessions import InMemorySessionRepository
from atlasai.application.threads import InMemoryThreadRepository, ThreadRecord
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
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
)

ingestion_events_table = Table(
    "ingestion_events",
    metadata,
    Column("job_id", String, nullable=False, index=True),
    Column("event_id", Integer, nullable=False),
    Column("payload", JSON, nullable=False),
    UniqueConstraint("job_id", "event_id", name="uq_ingestion_event_job_id"),
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
            questions_used=row["questions_used"],
            uploads_used=row["uploads_used"],
            active_ingestions=row["active_ingestions"],
            active_agent_runs=row["active_agent_runs"],
        )

    def get_request_key_snapshot(
        self,
        *,
        request_key: str,
    ) -> RequestKeyQuotaSnapshot:
        with self.engine.begin() as conn:
            row = self._get_or_create_request_key_quota(conn, request_key=request_key)
        return RequestKeyQuotaSnapshot(
            questions_used=row["questions_used"],
            uploads_used=row["uploads_used"],
        )

    def claim_question(
        self,
        *,
        user_id: str,
        request_key: str,
        user_limit: int,
        request_key_limit: int,
        concurrency_limit: int,
    ) -> AdmissionResult:
        with self.engine.begin() as conn:
            user_row = self._get_or_create_user_quota(conn, user_id=user_id, lock=True)
            request_key_row = self._get_or_create_request_key_quota(
                conn,
                request_key=request_key,
                lock=True,
            )

            if user_row["questions_used"] >= user_limit:
                return AdmissionResult(False, "Question quota exceeded.")
            if request_key_row["questions_used"] >= request_key_limit:
                return AdmissionResult(
                    False,
                    "Question quota exceeded for this request key.",
                )
            if user_row["active_agent_runs"] >= concurrency_limit:
                return AdmissionResult(
                    False,
                    "Another agent run is already in progress.",
                )

            conn.execute(
                update(user_quotas_table)
                .where(user_quotas_table.c.user_id == user_id)
                .values(
                    questions_used=user_row["questions_used"] + 1,
                    active_agent_runs=user_row["active_agent_runs"] + 1,
                )
            )
            conn.execute(
                update(request_key_quotas_table)
                .where(request_key_quotas_table.c.request_key == request_key)
                .values(questions_used=request_key_row["questions_used"] + 1)
            )
        return AdmissionResult(True)

    def claim_upload(
        self,
        *,
        user_id: str,
        request_key: str,
        user_limit: int,
        request_key_limit: int,
        concurrency_limit: int,
    ) -> AdmissionResult:
        with self.engine.begin() as conn:
            user_row = self._get_or_create_user_quota(conn, user_id=user_id, lock=True)
            request_key_row = self._get_or_create_request_key_quota(
                conn,
                request_key=request_key,
                lock=True,
            )

            if user_row["uploads_used"] >= user_limit:
                return AdmissionResult(False, "Upload quota exceeded.")
            if request_key_row["uploads_used"] >= request_key_limit:
                return AdmissionResult(
                    False,
                    "Upload quota exceeded for this request key.",
                )
            if user_row["active_ingestions"] >= concurrency_limit:
                return AdmissionResult(
                    False,
                    "Another document ingestion is already in progress.",
                )

            conn.execute(
                update(user_quotas_table)
                .where(user_quotas_table.c.user_id == user_id)
                .values(
                    uploads_used=user_row["uploads_used"] + 1,
                    active_ingestions=user_row["active_ingestions"] + 1,
                )
            )
            conn.execute(
                update(request_key_quotas_table)
                .where(request_key_quotas_table.c.request_key == request_key)
                .values(uploads_used=request_key_row["uploads_used"] + 1)
            )
        return AdmissionResult(True)

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
                    created_at=job.created_at,
                    expires_at=job.expires_at,
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

    def append_event(self, *, job_id: str, payload: dict) -> None:
        with self.engine.begin() as conn:
            self._append_event(conn, job_id=job_id, payload=payload)

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
                .values(state="ready")
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
                .values(state="failed")
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
        self.documents._active_document_by_user.pop(user_id, None)
        self.quotas._snapshots.pop(user_id, None)
        self.usage._records.pop(user_id, None)
        self.usage._seen_run_ids.pop(user_id, None)


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


def build_in_memory_repository_bundle() -> InMemoryRepositoryBundle:
    return InMemoryRepositoryBundle(
        sessions=InMemorySessionRepository(),
        documents=InMemoryDocumentRepository(),
        quotas=InMemoryQuotaRepository(),
        threads=InMemoryThreadRepository(),
        usage=InMemoryUsageRepository(),
    )
