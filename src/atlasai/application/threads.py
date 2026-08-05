from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Protocol
from uuid import uuid4


@dataclass(frozen=True)
class ThreadRecord:
    """Server-owned thread record."""

    thread_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    document_id: str | None = None


class ThreadRepository(Protocol):
    """Thread record access."""

    def create_thread(
        self,
        *,
        user_id: str,
        expires_at: datetime,
        document_id: str | None = None,
    ) -> ThreadRecord: ...

    def get_thread(self, *, thread_id: str) -> ThreadRecord | None: ...


class InMemoryThreadRepository:
    """In-memory thread store."""

    def __init__(self) -> None:
        self._threads: dict[str, ThreadRecord] = {}
        self._lock = Lock()

    def create_thread(
        self,
        *,
        user_id: str,
        expires_at: datetime,
        document_id: str | None = None,
    ) -> ThreadRecord:
        with self._lock:
            thread = ThreadRecord(
                thread_id=str(uuid4()),
                user_id=user_id,
                created_at=datetime.now(expires_at.tzinfo),
                expires_at=expires_at,
                document_id=document_id,
            )
            self._threads[thread.thread_id] = thread
            return thread

    def get_thread(self, *, thread_id: str) -> ThreadRecord | None:
        with self._lock:
            return self._threads.get(thread_id)


class ThreadService:
    """Creates and resolves owned threads."""

    def __init__(self, repository: ThreadRepository) -> None:
        self.repository = repository

    def create_thread(
        self,
        *,
        user_id: str,
        expires_at: datetime,
        document_id: str | None = None,
    ) -> ThreadRecord:
        return self.repository.create_thread(
            user_id=user_id,
            expires_at=expires_at,
            document_id=document_id,
        )

    def get_owned_thread(self, *, user_id: str, thread_id: str) -> ThreadRecord | None:
        thread = self.repository.get_thread(thread_id=thread_id)
        if thread is None or thread.user_id != user_id:
            return None
        return thread
