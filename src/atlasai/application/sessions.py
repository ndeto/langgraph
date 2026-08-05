import base64
import hashlib
import hmac
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import uuid4

from atlasai.domain.models import AnonymousSession


@dataclass(frozen=True)
class SessionCookieSettings:
    """Signed session cookie settings."""

    name: str
    secret: str
    max_age_seconds: int
    secure: bool
    same_site: str = "lax"
    path: str = "/"


@dataclass(frozen=True)
class SessionResolution:
    """Resolved anonymous session."""

    session: AnonymousSession
    set_cookie: bool
    cookie_value: str


@dataclass
class CleanupJobRecord:
    """Queued cleanup work for a rotated session."""

    job_id: str
    user_id: str
    state: str
    created_at: datetime
    updated_at: datetime
    attempts: int = 0
    heartbeat_at: datetime | None = None
    failure_text: str | None = None


class SessionRepository(Protocol):
    """Session record access."""

    def get_session(self, *, user_id: str) -> AnonymousSession | None: ...

    def upsert_session(self, session: AnonymousSession) -> None: ...

    def expire_session(self, *, user_id: str) -> None: ...

    def enqueue_cleanup_job(self, *, user_id: str) -> CleanupJobRecord: ...

    def claim_next_cleanup_job(
        self,
        *,
        heartbeat_timeout_seconds: int,
        max_attempts: int,
    ) -> CleanupJobRecord | None: ...

    def heartbeat_cleanup_job(self, *, job_id: str) -> None: ...

    def mark_cleanup_job_ready(self, *, job_id: str) -> None: ...

    def mark_cleanup_job_failed(self, *, job_id: str, failure_text: str) -> None: ...


class InMemorySessionRepository:
    """In-memory session store."""

    def __init__(self) -> None:
        self._sessions: dict[str, AnonymousSession] = {}
        self._cleanup_jobs: dict[str, CleanupJobRecord] = {}

    def get_session(self, *, user_id: str) -> AnonymousSession | None:
        session = self._sessions.get(user_id)
        if session is None or session.expires_at <= datetime.now(UTC):
            return None
        return session

    def upsert_session(self, session: AnonymousSession) -> None:
        self._sessions[session.user_id] = session

    def expire_session(self, *, user_id: str) -> None:
        self._sessions.pop(user_id, None)

    def enqueue_cleanup_job(self, *, user_id: str) -> CleanupJobRecord:
        now = datetime.now(UTC)
        job = CleanupJobRecord(
            job_id=str(uuid4()),
            user_id=user_id,
            state="queued",
            created_at=now,
            updated_at=now,
        )
        self._cleanup_jobs[job.job_id] = job
        return job

    def claim_next_cleanup_job(
        self,
        *,
        heartbeat_timeout_seconds: int,
        max_attempts: int,
    ) -> CleanupJobRecord | None:
        stale_before = datetime.now(UTC) - timedelta(seconds=heartbeat_timeout_seconds)
        for job in self._cleanup_jobs.values():
            if job.state == "queued" or (
                job.state == "processing"
                and job.heartbeat_at is not None
                and job.heartbeat_at < stale_before
                and job.attempts < max_attempts
            ):
                now = datetime.now(UTC)
                claimed = CleanupJobRecord(
                    job_id=job.job_id,
                    user_id=job.user_id,
                    state="processing",
                    created_at=job.created_at,
                    updated_at=now,
                    attempts=job.attempts + 1,
                    heartbeat_at=now,
                    failure_text=None,
                )
                self._cleanup_jobs[job.job_id] = claimed
                return claimed
        return None

    def heartbeat_cleanup_job(self, *, job_id: str) -> None:
        job = self._cleanup_jobs[job_id]
        self._cleanup_jobs[job_id] = CleanupJobRecord(
            job_id=job.job_id,
            user_id=job.user_id,
            state=job.state,
            created_at=job.created_at,
            updated_at=datetime.now(UTC),
            attempts=job.attempts,
            heartbeat_at=datetime.now(UTC),
            failure_text=job.failure_text,
        )

    def mark_cleanup_job_ready(self, *, job_id: str) -> None:
        job = self._cleanup_jobs[job_id]
        self._cleanup_jobs[job_id] = CleanupJobRecord(
            job_id=job.job_id,
            user_id=job.user_id,
            state="ready",
            created_at=job.created_at,
            updated_at=datetime.now(UTC),
            attempts=job.attempts,
            heartbeat_at=job.heartbeat_at,
            failure_text=None,
        )

    def mark_cleanup_job_failed(self, *, job_id: str, failure_text: str) -> None:
        job = self._cleanup_jobs[job_id]
        self._cleanup_jobs[job_id] = CleanupJobRecord(
            job_id=job.job_id,
            user_id=job.user_id,
            state="failed",
            created_at=job.created_at,
            updated_at=datetime.now(UTC),
            attempts=job.attempts,
            heartbeat_at=job.heartbeat_at,
            failure_text=failure_text,
        )


class SessionService:
    """Issues and restores anonymous sessions."""

    def __init__(
        self,
        settings: SessionCookieSettings,
        repository: SessionRepository | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository

    def resolve_session(self, cookie_value: str | None) -> SessionResolution:
        cookie_session = self._deserialize_cookie(cookie_value) if cookie_value else None
        session = self._resolve_persisted_session(cookie_session)
        if session is not None:
            next_cookie_value = (
                self._serialize_cookie(session)
                if cookie_session != session or not cookie_value
                else cookie_value or ""
            )
            return SessionResolution(
                session=session,
                set_cookie=next_cookie_value != (cookie_value or ""),
                cookie_value=next_cookie_value,
            )

        new_session = self._new_session()
        return SessionResolution(
            session=new_session,
            set_cookie=True,
            cookie_value=self._serialize_cookie(new_session),
        )

    def serialize_session(self, session: AnonymousSession) -> str:
        """Serialize session state for cookie storage."""

        return self._serialize_cookie(session)

    def resolve_existing_session(
        self,
        cookie_value: str | None,
    ) -> AnonymousSession | None:
        """Resolve only a currently valid signed session without creating a new one."""

        if not cookie_value:
            return None

        cookie_session = self._deserialize_cookie(cookie_value)
        return self._resolve_persisted_session(cookie_session)

    def rotate_session(
        self,
        current_session: AnonymousSession | None,
    ) -> SessionResolution:
        """Expire the current session and issue a fresh anonymous session."""

        if current_session is not None and self.repository is not None:
            self.repository.expire_session(user_id=current_session.user_id)

        new_session = self._new_session()
        return SessionResolution(
            session=new_session,
            set_cookie=True,
            cookie_value=self._serialize_cookie(new_session),
        )

    def with_active_thread(
        self,
        session: AnonymousSession,
        thread_id: str | None,
    ) -> AnonymousSession:
        """Return session state with the active thread set."""

        updated_session = replace(session, active_thread=thread_id)
        self._persist_session(updated_session)
        return updated_session

    def with_active_document(
        self,
        session: AnonymousSession,
        document_id: str | None,
    ) -> AnonymousSession:
        """Return session state with the active document set."""

        updated_session = replace(session, active_document=document_id)
        self._persist_session(updated_session)
        return updated_session

    def _new_session(self) -> AnonymousSession:
        expires_at = datetime.now(UTC) + timedelta(seconds=self.settings.max_age_seconds)
        session = AnonymousSession(
            user_id=str(uuid4()),
            expires_at=expires_at,
        )
        self._persist_session(session)
        return session

    def _resolve_persisted_session(
        self,
        cookie_session: AnonymousSession | None,
    ) -> AnonymousSession | None:
        if cookie_session is None:
            return None
        if self.repository is None:
            return cookie_session

        persisted_session = self.repository.get_session(user_id=cookie_session.user_id)
        return persisted_session

    def _persist_session(self, session: AnonymousSession) -> None:
        if self.repository is not None:
            self.repository.upsert_session(session)

    def _serialize_cookie(self, session: AnonymousSession) -> str:
        payload = json.dumps(
            {
                "user_id": session.user_id,
                "expires_at": session.expires_at.isoformat(),
                "active_document": session.active_document,
                "active_thread": session.active_thread,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        encoded_payload = base64.urlsafe_b64encode(payload).decode("ascii")
        signature = self._sign(encoded_payload)
        return f"{encoded_payload}.{signature}"

    def _deserialize_cookie(self, cookie_value: str) -> AnonymousSession | None:
        try:
            encoded_payload, signature = cookie_value.rsplit(".", 1)
        except ValueError:
            return None

        expected_signature = self._sign(encoded_payload)
        if not hmac.compare_digest(signature, expected_signature):
            return None

        try:
            payload = base64.urlsafe_b64decode(encoded_payload.encode("ascii"))
            data = json.loads(payload.decode("utf-8"))
            user_id = data["user_id"]
            expires_at = datetime.fromisoformat(data["expires_at"])
            active_document = data.get("active_document")
            active_thread = data.get("active_thread")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)

        if expires_at <= datetime.now(UTC):
            return None

        return AnonymousSession(
            user_id=user_id,
            expires_at=expires_at,
            active_document=active_document
            if isinstance(active_document, str)
            else None,
            active_thread=active_thread if isinstance(active_thread, str) else None,
        )

    def _sign(self, payload: str) -> str:
        digest = hmac.new(
            self.settings.secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        )
        return digest.hexdigest()
