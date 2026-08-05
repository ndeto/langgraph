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


@dataclass(frozen=True)
class ThreadMessageRecord:
    """User-visible persisted thread message."""

    message_id: str
    thread_id: str
    role: str
    content: str
    created_at: datetime
    assets: list[dict[str, str]] | None = None
    status: str = "done"


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

    def append_message(
        self,
        *,
        thread_id: str,
        role: str,
        content: str,
        assets: list[dict[str, str]] | None = None,
        status: str = "done",
    ) -> ThreadMessageRecord: ...

    def list_messages(self, *, thread_id: str) -> list[ThreadMessageRecord]: ...


class InMemoryThreadRepository:
    """In-memory thread store."""

    def __init__(self) -> None:
        self._threads: dict[str, ThreadRecord] = {}
        self._messages_by_thread: dict[str, list[ThreadMessageRecord]] = {}
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

    def append_message(
        self,
        *,
        thread_id: str,
        role: str,
        content: str,
        assets: list[dict[str, str]] | None = None,
        status: str = "done",
    ) -> ThreadMessageRecord:
        with self._lock:
            thread = self._threads[thread_id]
            message = ThreadMessageRecord(
                message_id=str(uuid4()),
                thread_id=thread_id,
                role=role,
                content=content,
                created_at=datetime.now(thread.expires_at.tzinfo),
                assets=assets,
                status=status,
            )
            self._messages_by_thread.setdefault(thread_id, []).append(message)
            return message

    def list_messages(self, *, thread_id: str) -> list[ThreadMessageRecord]:
        with self._lock:
            return list(self._messages_by_thread.get(thread_id, []))


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

    def append_owned_message(
        self,
        *,
        user_id: str,
        thread_id: str,
        role: str,
        content: str,
        assets: list[dict[str, str]] | None = None,
        status: str = "done",
    ) -> ThreadMessageRecord | None:
        thread = self.get_owned_thread(user_id=user_id, thread_id=thread_id)
        if thread is None:
            return None
        return self.repository.append_message(
            thread_id=thread_id,
            role=role,
            content=content,
            assets=assets,
            status=status,
        )

    def list_owned_messages(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> list[ThreadMessageRecord] | None:
        thread = self.get_owned_thread(user_id=user_id, thread_id=thread_id)
        if thread is None:
            return None
        return self.repository.list_messages(thread_id=thread_id)
