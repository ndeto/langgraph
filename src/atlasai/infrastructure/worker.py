import asyncio
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from atlasai.application.documents import DocumentService
from atlasai.infrastructure.postgres_repositories import (
    InMemoryRepositoryBundle,
    PostgresRepositoryBundle,
)
from atlasai.infrastructure.telemetry import usage_record_from_callback
from atlasai.rag.image_payloads import IMAGE_OUTPUT_DIR
from atlasai.store.hybrid_store import PostgresVectorService


@dataclass(frozen=True)
class WorkerSettings:
    """Polling and retry settings for the background worker."""

    poll_seconds: float
    cleanup_poll_seconds: float
    heartbeat_timeout_seconds: int
    max_attempts: int


def load_worker_settings() -> WorkerSettings:
    return WorkerSettings(
        poll_seconds=float(os.getenv("ATLASAI_WORKER_POLL_SECONDS", "1")),
        cleanup_poll_seconds=float(
            os.getenv("ATLASAI_WORKER_CLEANUP_POLL_SECONDS", "15")
        ),
        heartbeat_timeout_seconds=int(
            os.getenv("ATLASAI_WORKER_HEARTBEAT_TIMEOUT_SECONDS", "60")
        ),
        max_attempts=int(os.getenv("ATLASAI_WORKER_MAX_ATTEMPTS", "3")),
    )


def _document_service(
    repositories: PostgresRepositoryBundle | InMemoryRepositoryBundle,
) -> DocumentService:
    if repositories.documents is None:
        raise RuntimeError("Repository bundle startup must run before use.")
    return DocumentService(repository=repositories.documents)


async def process_next_ingestion_job(
    repositories: PostgresRepositoryBundle | InMemoryRepositoryBundle,
    vector_service: PostgresVectorService,
    *,
    settings: WorkerSettings,
) -> bool:
    document_service = _document_service(repositories)
    job = document_service.claim_next_job(
        heartbeat_timeout_seconds=settings.heartbeat_timeout_seconds,
        max_attempts=settings.max_attempts,
    )
    if job is None:
        return False

    if repositories.quotas is None or repositories.usage is None:
        raise RuntimeError("Repository bundle startup must run before use.")

    try:
        document = repositories.documents.get_document(document_id=job.document_id)
        if document is None or not job.source_path:
            repositories.documents.mark_failed(
                job_id=job.job_id,
                text="Missing document staging path.",
            )
            return True

        from atlasai.rag.rag_ingestion import stream_ingest_pdf

        await document_service.run_job(
            job_id=job.job_id,
            file_path=job.source_path,
            file_name=document.filename,
            user_id=job.user_id,
            document_id=job.document_id,
            stream_ingest_pdf=stream_ingest_pdf,
            storage=vector_service.get_store("raggidy_docs"),
            progress_callback=lambda claimed_job_id: document_service.heartbeat_job(
                job_id=claimed_job_id
            ),
        )
        return True
    finally:
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
        shutil.rmtree(IMAGE_OUTPUT_DIR / job.user_id, ignore_errors=True)
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
    while True:
        processed = await process_next_cleanup_job(
            repositories,
            vector_service,
            settings=settings,
        )
        if not processed:
            await asyncio.sleep(settings.cleanup_poll_seconds)


async def run_worker() -> None:
    repositories = PostgresRepositoryBundle()
    vector_service = PostgresVectorService()
    settings = load_worker_settings()

    await repositories.startup()
    await vector_service.startup()

    try:
        await asyncio.gather(
            _run_ingestion_loop(repositories, vector_service, settings),
            _run_cleanup_loop(repositories, vector_service, settings),
        )
    finally:
        await vector_service.shutdown()
        await repositories.shutdown()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
