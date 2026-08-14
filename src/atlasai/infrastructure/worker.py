import asyncio
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Protocol, cast

from atlasai.application.documents import DocumentService
from atlasai.infrastructure.postgres_repositories import (
    InMemoryRepositoryBundle,
    PostgresRepositoryBundle,
)
from atlasai.infrastructure.telemetry import usage_record_from_callback
from atlasai.lib.logging import configure_logging
from atlasai.store.hybrid_store import PostgresVectorService

configure_logging()

logger = logging.getLogger(__name__)


def terminal_ingestion_failure_message(max_attempts: int) -> str:
    """Return the user-facing message for permanently failed ingestion jobs."""

    return (
        f"Document processing failed after {max_attempts} attempts. "
        "Please try a smaller or simpler PDF."
    )


class ExpiredSessionCleanupRepository(Protocol):
    """Optional repository hook for discovering expired sessions."""

    def enqueue_expired_session_cleanup_jobs(self) -> int: ...


@dataclass(frozen=True)
class WorkerSettings:
    """Polling and retry settings for the background worker."""

    poll_seconds: float
    cleanup_poll_seconds: float
    heartbeat_timeout_seconds: int
    max_attempts: int
    heartbeat_seconds: float = 15.0


def load_worker_settings() -> WorkerSettings:
    return WorkerSettings(
        poll_seconds=float(os.getenv("ATLASAI_WORKER_POLL_SECONDS", "1")),
        cleanup_poll_seconds=float(
            os.getenv("ATLASAI_WORKER_CLEANUP_POLL_SECONDS", "15")
        ),
        heartbeat_timeout_seconds=int(
            os.getenv("ATLASAI_WORKER_HEARTBEAT_TIMEOUT_SECONDS", "120")
        ),
        max_attempts=int(os.getenv("ATLASAI_WORKER_MAX_ATTEMPTS", "3")),
        heartbeat_seconds=float(
            os.getenv("ATLASAI_WORKER_HEARTBEAT_SECONDS", "15")
        ),
    )


def _document_service(
    repositories: PostgresRepositoryBundle | InMemoryRepositoryBundle,
) -> DocumentService:
    if repositories.documents is None:
        raise RuntimeError("Repository bundle startup must run before use.")
    return DocumentService(repository=repositories.documents)


def _heartbeat_ingestion_job(
    document_service: DocumentService,
    job_id: str,
    stop_event: Event,
    heartbeat_seconds: float,
) -> None:
    while not stop_event.wait(heartbeat_seconds):
        try:
            document_service.heartbeat_job(job_id=job_id)
        except Exception:
            logger.exception("Ingestion heartbeat failed job_id=%s", job_id)
            return


@contextmanager
def _maintain_ingestion_heartbeat(
    document_service: DocumentService,
    *,
    job_id: str,
    heartbeat_seconds: float,
):
    """Maintain a heartbeat even while synchronous PDF work blocks asyncio."""

    stop_event = Event()
    thread = Thread(
        target=_heartbeat_ingestion_job,
        args=(document_service, job_id, stop_event, heartbeat_seconds),
        name=f"atlasai-ingestion-heartbeat-{job_id}",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()


def enqueue_expired_session_cleanup_jobs(
    repositories: PostgresRepositoryBundle | InMemoryRepositoryBundle,
) -> int:
    """Queue expired sessions when the repository provides discovery support."""

    enqueue = getattr(repositories, "enqueue_expired_session_cleanup_jobs", None)
    if not callable(enqueue):
        return 0
    repository = cast(ExpiredSessionCleanupRepository, repositories)
    return repository.enqueue_expired_session_cleanup_jobs()


async def fail_stale_exhausted_ingestion_job(
    repositories: PostgresRepositoryBundle | InMemoryRepositoryBundle,
    vector_service: PostgresVectorService,
    *,
    settings: WorkerSettings,
) -> bool:
    """Mark exhausted stale ingestion jobs failed and release their resources."""

    document_service = _document_service(repositories)
    job = document_service.repository.fail_stale_exhausted_job(
        heartbeat_timeout_seconds=settings.heartbeat_timeout_seconds,
        max_attempts=settings.max_attempts,
        text=terminal_ingestion_failure_message(settings.max_attempts),
    )
    if job is None:
        return False

    logger.error(
        "worker_ingestion_exhausted job_id=%s user_id=%s document_id=%s attempts=%s",
        job.job_id,
        job.user_id,
        job.document_id,
        job.attempts,
    )

    await vector_service.adelete_by_document(
        table_name="raggidy_docs",
        user_id=job.user_id,
        document_id=job.document_id,
    )

    if isinstance(repositories, PostgresRepositoryBundle) and repositories.assets is not None:
        repositories.assets.delete_document_assets(
            document_id=job.document_id,
            user_id=job.user_id,
        )

    if job.source_path:
        Path(job.source_path).unlink(missing_ok=True)

    if repositories.usage is None or repositories.quotas is None:
        raise RuntimeError("Repository bundle startup must run before use.")

    repositories.usage.append_record(
        user_id=job.user_id,
        record=usage_record_from_callback(
            operation="ingestion",
            payload=None,
            run_id=job.job_id,
        ),
    )
    repositories.quotas.release_ingestion(user_id=job.user_id)
    return True


async def process_next_ingestion_job(
    repositories: PostgresRepositoryBundle | InMemoryRepositoryBundle,
    vector_service: PostgresVectorService,
    *,
    settings: WorkerSettings,
) -> bool:
    document_service = _document_service(repositories)
    if repositories.quotas is None or repositories.usage is None:
        raise RuntimeError("Repository bundle startup must run before use.")

    if await fail_stale_exhausted_ingestion_job(
        repositories,
        vector_service,
        settings=settings,
    ):
        return True

    job = document_service.claim_next_job(
        heartbeat_timeout_seconds=settings.heartbeat_timeout_seconds,
        max_attempts=settings.max_attempts,
    )
    if job is None:
        return False

    logger.info(
        "worker_claimed_ingestion_job job_id=%s user_id=%s document_id=%s attempts=%s source_path=%s",
        job.job_id,
        job.user_id,
        job.document_id,
        job.attempts,
        job.source_path,
    )

    asset_repository = (
        repositories.assets
        if isinstance(repositories, PostgresRepositoryBundle)
        else None
    )
    terminal = False

    try:
        document = document_service.repository.get_document(
            document_id=job.document_id
        )
        if document is None or not job.source_path:
            logger.error(
                "worker_missing_document_or_source job_id=%s user_id=%s document_id=%s source_path=%s document_found=%s",
                job.job_id,
                job.user_id,
                job.document_id,
                job.source_path,
                document is not None,
            )
            document_service.repository.mark_failed(
                job_id=job.job_id,
                text="Missing document staging path.",
            )
            terminal = True
            return True

        if job.attempts > 1:
            await vector_service.adelete_by_document(
                table_name="raggidy_docs",
                user_id=job.user_id,
                document_id=job.document_id,
            )
            if asset_repository is not None:
                asset_repository.delete_document_assets(
                    document_id=job.document_id,
                    user_id=job.user_id,
                )

        from atlasai.rag.rag_ingestion import stream_ingest_pdf

        logger.info(
            "worker_starting_ingestion job_id=%s user_id=%s document_id=%s source_path=%s filename=%s",
            job.job_id,
            job.user_id,
            job.document_id,
            job.source_path,
            document.filename,
        )
        with _maintain_ingestion_heartbeat(
            document_service,
            job_id=job.job_id,
            heartbeat_seconds=settings.heartbeat_seconds,
        ):
            await document_service.run_job(
                job_id=job.job_id,
                file_path=job.source_path,
                file_name=document.filename,
                user_id=job.user_id,
                document_id=job.document_id,
                stream_ingest_pdf=stream_ingest_pdf,
                storage=vector_service.get_store("raggidy_docs"),
                asset_repository=asset_repository,
            )
        if job.client_key and job.ip_hash:
            repositories.quotas.complete_upload(
                client_key=job.client_key,
                ip_hash=job.ip_hash,
            )
        terminal = True
        logger.info(
            "worker_completed_ingestion job_id=%s user_id=%s document_id=%s",
            job.job_id,
            job.user_id,
            job.document_id,
        )
        return True
    except Exception as exc:
        logger.exception(
            "worker_ingestion_failed job_id=%s user_id=%s document_id=%s error_type=%s error=%s",
            job.job_id,
            job.user_id,
            job.document_id,
            type(exc).__name__,
            exc,
        )
        if job.attempts < settings.max_attempts:
            document_service.retry_job(job_id=job.job_id, text=str(exc))
        else:
            document_service.repository.mark_failed(
                job_id=job.job_id,
                text=terminal_ingestion_failure_message(settings.max_attempts),
            )
            terminal = True
        return True
    finally:
        if terminal:
            if job.source_path:
                Path(job.source_path).unlink(missing_ok=True)
            repositories.usage.append_record(
                user_id=job.user_id,
                record=usage_record_from_callback(
                    operation="ingestion",
                    payload=None,
                    run_id=job.job_id,
                ),
            )
            repositories.quotas.release_ingestion(user_id=job.user_id)


async def process_next_cleanup_job(
    repositories: PostgresRepositoryBundle | InMemoryRepositoryBundle,
    vector_service: PostgresVectorService,
    *,
    settings: WorkerSettings,
) -> bool:
    job = repositories.claim_next_cleanup_job(
        heartbeat_timeout_seconds=settings.heartbeat_timeout_seconds,
        max_attempts=settings.max_attempts,
    )
    if job is None:
        return False

    document_service = _document_service(repositories)

    try:
        for source_path in document_service.list_user_source_paths(user_id=job.user_id):
            Path(source_path).unlink(missing_ok=True)
        repositories.heartbeat_cleanup_job(job_id=job.job_id)

        await vector_service.adelete_by_user(
            table_name="raggidy_docs",
            user_id=job.user_id,
        )
        repositories.heartbeat_cleanup_job(job_id=job.job_id)

        repositories.delete_user_data(user_id=job.user_id)
        repositories.mark_cleanup_job_ready(job_id=job.job_id)
        return True
    except Exception as exc:
        repositories.mark_cleanup_job_failed(
            job_id=job.job_id,
            failure_text=str(exc),
        )
        return True


async def _run_ingestion_loop(
    repositories: PostgresRepositoryBundle | InMemoryRepositoryBundle,
    vector_service: PostgresVectorService,
    settings: WorkerSettings,
) -> None:
    while True:
        processed = await process_next_ingestion_job(
            repositories,
            vector_service,
            settings=settings,
        )
        if not processed:
            await asyncio.sleep(settings.poll_seconds)


async def _run_cleanup_loop(
    repositories: PostgresRepositoryBundle | InMemoryRepositoryBundle,
    vector_service: PostgresVectorService,
    settings: WorkerSettings,
) -> None:
    loop = asyncio.get_running_loop()
    next_expiry_sweep = 0.0
    while True:
        now = loop.time()
        if now >= next_expiry_sweep:
            enqueue_expired_session_cleanup_jobs(repositories)
            next_expiry_sweep = now + settings.cleanup_poll_seconds

        processed = await process_next_cleanup_job(
            repositories,
            vector_service,
            settings=settings,
        )
        if not processed:
            await asyncio.sleep(settings.cleanup_poll_seconds)


async def run_worker() -> None:
    logger.info("worker_booting")
    repositories = PostgresRepositoryBundle()
    vector_service = PostgresVectorService()
    settings = load_worker_settings()

    await repositories.startup()
    await vector_service.startup()
    logger.info(
        "worker_ready poll_seconds=%s cleanup_poll_seconds=%s heartbeat_timeout_seconds=%s max_attempts=%s",
        settings.poll_seconds,
        settings.cleanup_poll_seconds,
        settings.heartbeat_timeout_seconds,
        settings.max_attempts,
    )

    try:
        await asyncio.gather(
            _run_ingestion_loop(repositories, vector_service, settings),
            _run_cleanup_loop(repositories, vector_service, settings),
        )
    finally:
        logger.info("worker_shutting_down")
        await vector_service.shutdown()
        await repositories.shutdown()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
