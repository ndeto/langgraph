import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Protocol
from uuid import uuid4


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
    ) -> tuple[DocumentRecord, IngestionJobRecord]: ...

    def get_document(self, *, document_id: str) -> DocumentRecord | None: ...

    def get_job(self, *, job_id: str) -> IngestionJobRecord | None: ...

    def get_active_document(self, *, user_id: str) -> DocumentRecord | None: ...

    def append_event(self, *, job_id: str, payload: dict) -> None: ...

    def mark_ready(self, *, job_id: str) -> None: ...

    def mark_failed(self, *, job_id: str, text: str) -> None: ...


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
            self._append_event_locked(
                job_id=job_id,
                payload={"type": "failed", "text": text},
            )

    def _append_event_locked(self, *, job_id: str, payload: dict) -> None:
        job = self._jobs[job_id]
        job.events.append(
            IngestionEventRecord(event_id=len(job.events) + 1, payload=payload)
        )

    def _set_job_state_locked(self, *, job_id: str, state: str) -> None:
        job = self._jobs[job_id]
        job.state = state
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
    ) -> tuple[DocumentRecord, IngestionJobRecord]:
        return self.repository.create_job(
            user_id=user_id,
            filename=filename,
            size_bytes=size_bytes,
            ttl_seconds=ttl_seconds,
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

    async def run_job(
        self,
        *,
        job_id: str,
        file_path: str,
        file_name: str,
        user_id: str | None = None,
        stream_ingest_pdf,
        storage=None,
    ) -> None:
        try:
            async for event in stream_ingest_pdf(
                file_path,
                file_name=file_name,
                user_id=user_id,
                storage=storage,
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
                await asyncio.sleep(0)
            self.repository.mark_ready(job_id=job_id)
        except Exception as exc:
            self.repository.mark_failed(job_id=job_id, text=str(exc))
        finally:
            path = Path(file_path)
            if path.exists():
                path.unlink()
