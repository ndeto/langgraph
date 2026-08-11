import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Protocol
from uuid import uuid4


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentRecord:
    """Owned document record."""

    document_id: str
    user_id: str
    filename: str
    size_bytes: int
    status: str
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True)
class IngestionEventRecord:
    """Persisted ingestion event."""

    event_id: int
    payload: dict


@dataclass
class IngestionJobRecord:
    """Owned ingestion job record."""

    job_id: str
    user_id: str
    document_id: str
    state: str
    created_at: datetime
    expires_at: datetime
    source_path: str | None = None
    attempts: int = 0
    heartbeat_at: datetime | None = None
    updated_at: datetime | None = None
    failure_text: str | None = None
    client_key: str | None = None
    ip_hash: str | None = None
    events: list[IngestionEventRecord] = field(default_factory=list)


class DocumentRepository(Protocol):
    """Document record access."""

    def create_job(
        self,
        *,
        user_id: str,
        filename: str,
        size_bytes: int,
        ttl_seconds: int,
        source_path: str,
        client_key: str | None = None,
        ip_hash: str | None = None,
    ) -> tuple[DocumentRecord, IngestionJobRecord]: ...

    def get_document(self, *, document_id: str) -> DocumentRecord | None: ...

    def get_job(self, *, job_id: str) -> IngestionJobRecord | None: ...

    def get_active_document(self, *, user_id: str) -> DocumentRecord | None: ...

    def list_documents(self, *, user_id: str) -> list[DocumentRecord]: ...

    def list_user_source_paths(self, *, user_id: str) -> list[str]: ...

    def append_event(self, *, job_id: str, payload: dict) -> None: ...

    def mark_ready(self, *, job_id: str) -> None: ...

    def mark_failed(self, *, job_id: str, text: str) -> None: ...

    def mark_retry(self, *, job_id: str, text: str) -> None: ...

    def claim_next_job(
        self,
        *,
        heartbeat_timeout_seconds: int,
        max_attempts: int,
    ) -> IngestionJobRecord | None: ...

    def heartbeat_job(self, *, job_id: str) -> None: ...


class InMemoryDocumentRepository:
    """In-memory document store."""

    def __init__(self) -> None:
        self._documents: dict[str, DocumentRecord] = {}
        self._jobs: dict[str, IngestionJobRecord] = {}
        self._active_document_by_user: dict[str, str] = {}
        self._lock = Lock()

    def create_job(
        self,
        *,
        user_id: str,
        filename: str,
        size_bytes: int,
        ttl_seconds: int,
        source_path: str,
        client_key: str | None = None,
        ip_hash: str | None = None,
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        with self._lock:
            created_at = datetime.now(UTC)
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
                client_key=client_key,
                ip_hash=ip_hash,
            )
            self._documents[document.document_id] = document
            self._jobs[job.job_id] = job
            self._append_event_locked(
                job_id=job.job_id,
                payload={"type": "queued", "text": "Document queued"},
            )
            return document, job

    def get_document(self, *, document_id: str) -> DocumentRecord | None:
        return self._documents.get(document_id)

    def get_job(self, *, job_id: str) -> IngestionJobRecord | None:
        return self._jobs.get(job_id)

    def get_active_document(self, *, user_id: str) -> DocumentRecord | None:
        document_id = self._active_document_by_user.get(user_id)
        if document_id is None:
            return None
        return self._documents.get(document_id)

    def list_documents(self, *, user_id: str) -> list[DocumentRecord]:
        return sorted(
            [
                document
                for document in self._documents.values()
                if document.user_id == user_id
            ],
            key=lambda document: document.created_at,
        )

    def list_user_source_paths(self, *, user_id: str) -> list[str]:
        return [
            job.source_path
            for job in self._jobs.values()
            if job.user_id == user_id and job.source_path
        ]

    def append_event(self, *, job_id: str, payload: dict) -> None:
        with self._lock:
            self._append_event_locked(job_id=job_id, payload=payload)

    def mark_ready(self, *, job_id: str) -> None:
        with self._lock:
            self._set_job_state_locked(job_id=job_id, state="ready")
            job = self._jobs[job_id]
            self._active_document_by_user[job.user_id] = job.document_id
            self._append_event_locked(
                job_id=job_id,
                payload={"type": "ready", "text": "Document ready"},
            )

    def mark_failed(self, *, job_id: str, text: str) -> None:
        with self._lock:
            self._set_job_state_locked(job_id=job_id, state="failed")
            self._jobs[job_id].failure_text = text
            self._append_event_locked(
                job_id=job_id,
                payload={"type": "failed", "text": text},
            )

    def mark_retry(self, *, job_id: str, text: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.state = "queued"
            job.heartbeat_at = None
            job.updated_at = datetime.now(UTC)
            job.failure_text = text
            self._append_event_locked(
                job_id=job_id,
                payload={"type": "processing", "text": "Retrying ingestion."},
            )

    def claim_next_job(
        self,
        *,
        heartbeat_timeout_seconds: int,
        max_attempts: int,
    ) -> IngestionJobRecord | None:
        with self._lock:
            stale_before = datetime.now(UTC) - timedelta(
                seconds=heartbeat_timeout_seconds
            )
            for job in self._jobs.values():
                if job.expires_at <= datetime.now(UTC):
                    continue
                if job.state == "queued" or (
                    job.state == "processing"
                    and job.heartbeat_at is not None
                    and job.heartbeat_at < stale_before
                    and job.attempts < max_attempts
                ):
                    now = datetime.now(UTC)
                    job.state = "processing"
                    job.attempts += 1
                    job.heartbeat_at = now
                    job.updated_at = now
                    job.failure_text = None
                    return job
        return None

    def heartbeat_job(self, *, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            now = datetime.now(UTC)
            job.heartbeat_at = now
            job.updated_at = now

    def _append_event_locked(self, *, job_id: str, payload: dict) -> None:
        job = self._jobs[job_id]
        job.events.append(
            IngestionEventRecord(event_id=len(job.events) + 1, payload=payload)
        )

    def _set_job_state_locked(self, *, job_id: str, state: str) -> None:
        job = self._jobs[job_id]
        job.state = state
        job.updated_at = datetime.now(UTC)
        document = self._documents[job.document_id]
        self._documents[job.document_id] = DocumentRecord(
            document_id=document.document_id,
            user_id=document.user_id,
            filename=document.filename,
            size_bytes=document.size_bytes,
            status=state,
            created_at=document.created_at,
            expires_at=document.expires_at,
        )


class DocumentService:
    """Owns document jobs and ownership checks."""

    def __init__(self, repository: DocumentRepository) -> None:
        self.repository = repository

    def create_document_job(
        self,
        *,
        user_id: str,
        filename: str,
        size_bytes: int,
        ttl_seconds: int,
        source_path: str,
        client_key: str | None = None,
        ip_hash: str | None = None,
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        return self.repository.create_job(
            user_id=user_id,
            filename=filename,
            size_bytes=size_bytes,
            ttl_seconds=ttl_seconds,
            source_path=source_path,
            client_key=client_key,
            ip_hash=ip_hash,
        )

    def get_owned_job(self, *, user_id: str, job_id: str) -> IngestionJobRecord | None:
        job = self.repository.get_job(job_id=job_id)
        if job is None or job.user_id != user_id:
            return None
        return job

    def get_active_document(self, *, user_id: str) -> DocumentRecord | None:
        document = self.repository.get_active_document(user_id=user_id)
        if document is None or document.user_id != user_id:
            return None
        return document

    def list_documents(self, *, user_id: str) -> list[DocumentRecord]:
        return [
            document
            for document in self.repository.list_documents(user_id=user_id)
            if document.user_id == user_id
        ]

    def list_user_source_paths(self, *, user_id: str) -> list[str]:
        return self.repository.list_user_source_paths(user_id=user_id)

    def claim_next_job(
        self,
        *,
        heartbeat_timeout_seconds: int,
        max_attempts: int,
    ) -> IngestionJobRecord | None:
        return self.repository.claim_next_job(
            heartbeat_timeout_seconds=heartbeat_timeout_seconds,
            max_attempts=max_attempts,
        )

    def heartbeat_job(self, *, job_id: str) -> None:
        self.repository.heartbeat_job(job_id=job_id)

    def retry_job(self, *, job_id: str, text: str) -> None:
        self.repository.mark_retry(job_id=job_id, text=text)

    async def run_job(
        self,
        *,
        job_id: str,
        file_path: str,
        file_name: str,
        user_id: str | None = None,
        document_id: str | None = None,
        stream_ingest_pdf,
        storage=None,
        asset_repository=None,
        progress_callback=None,
    ) -> None:
        resolved_path = Path(file_path)
        try:
            logger.info(
                "Starting ingestion job job_id=%s document_id=%s user_id=%s file_name=%s "
                "file_path=%s exists=%s size_bytes=%s",
                job_id,
                document_id,
                user_id,
                file_name,
                resolved_path,
                resolved_path.exists(),
                resolved_path.stat().st_size if resolved_path.exists() else "missing",
            )
            async for event in stream_ingest_pdf(
                file_path,
                file_name=file_name,
                user_id=user_id,
                document_id=document_id,
                storage=storage,
                asset_repository=asset_repository,
                store_name="raggidy_docs",
            ):
                payload = dict(event)
                payload["type"] = {
                    "file": "validating",
                    "log": "processing",
                    "stats": "storing",
                    "done": "ready",
                }.get(str(payload.get("type")), str(payload.get("type")))
                self.repository.append_event(job_id=job_id, payload=payload)
                if progress_callback is not None:
                    progress_callback(job_id)
                await asyncio.sleep(0)
            self.repository.mark_ready(job_id=job_id)
            logger.info(
                "Completed ingestion job job_id=%s document_id=%s user_id=%s",
                job_id,
                document_id,
                user_id,
            )
        except Exception as exc:
            logger.exception(
                "Ingestion job failed job_id=%s document_id=%s user_id=%s file_name=%s "
                "file_path=%s error_type=%s error=%s",
                job_id,
                document_id,
                user_id,
                file_name,
                resolved_path,
                type(exc).__name__,
                exc,
            )
            raise
