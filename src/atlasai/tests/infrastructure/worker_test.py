import asyncio
import tempfile
import time
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from atlasai.infrastructure.postgres_repositories import (
    build_in_memory_repository_bundle,
)
from atlasai.infrastructure.worker import (
    WorkerSettings,
    _maintain_ingestion_heartbeat,
    enqueue_expired_session_cleanup_jobs,
    load_worker_settings,
    process_next_ingestion_job,
)


class _HeartbeatDocumentService:
    def __init__(self) -> None:
        self.heartbeats = 0

    def heartbeat_job(self, *, job_id: str) -> None:
        del job_id
        self.heartbeats += 1


class _VectorService:
    def __init__(self) -> None:
        self.deleted_documents: list[tuple[str, str]] = []

    def get_store(self, _table_name: str):
        return None

    async def adelete_by_document(
        self, *, table_name: str, user_id: str, document_id: str
    ) -> None:
        del table_name
        self.deleted_documents.append((user_id, document_id))


class WorkerHardeningTest(unittest.TestCase):
    def test_worker_defaults_use_three_attempts_and_120_second_stale_timeout(self):
        settings = load_worker_settings()

        self.assertEqual(settings.max_attempts, 3)
        self.assertEqual(settings.heartbeat_timeout_seconds, 120)
        self.assertEqual(settings.heartbeat_seconds, 15.0)

    def test_ingestion_heartbeat_runs_independently(self):
        document_service = _HeartbeatDocumentService()

        with _maintain_ingestion_heartbeat(
            document_service,  # type: ignore[arg-type]
            job_id="job-1",
            heartbeat_seconds=0.01,
        ):
            time.sleep(0.04)

        self.assertGreaterEqual(document_service.heartbeats, 2)

    def test_expired_session_sweep_is_noop_without_repository_hook(self):
        repositories = build_in_memory_repository_bundle()

        queued = enqueue_expired_session_cleanup_jobs(repositories)

        self.assertEqual(queued, 0)

    def test_failed_ingestion_retries_before_removing_staged_pdf(self):
        repositories = build_in_memory_repository_bundle()
        vector_service = _VectorService()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as staged:
            staged.write(b"%PDF-1.4")
            source_path = staged.name
        document, job = repositories.documents.create_job(
            user_id="user-1",
            filename="sample.pdf",
            size_bytes=8,
            ttl_seconds=3600,
            source_path=source_path,
        )
        calls = 0

        async def stream_ingest_pdf(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary failure")
            yield {"type": "done", "text": "Ingestion complete"}

        settings = WorkerSettings(
            poll_seconds=0.01,
            cleanup_poll_seconds=0.01,
            heartbeat_timeout_seconds=60,
            max_attempts=3,
            heartbeat_seconds=0.01,
        )
        with patch(
            "atlasai.rag.rag_ingestion.stream_ingest_pdf",
            new=stream_ingest_pdf,
        ):
            asyncio.run(
                process_next_ingestion_job(
                    repositories,
                    vector_service,  # type: ignore[arg-type]
                    settings=settings,
                )
            )
            self.assertEqual(repositories.documents.get_job(job_id=job.job_id).state, "queued")
            self.assertTrue(Path(source_path).exists())

            asyncio.run(
                process_next_ingestion_job(
                    repositories,
                    vector_service,  # type: ignore[arg-type]
                    settings=settings,
                )
            )

        self.assertEqual(repositories.documents.get_job(job_id=job.job_id).state, "ready")
        self.assertEqual(vector_service.deleted_documents, [("user-1", document.document_id)])
        self.assertFalse(Path(source_path).exists())

    def test_stale_exhausted_ingestion_job_fails_permanently(self):
        repositories = build_in_memory_repository_bundle()
        vector_service = _VectorService()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as staged:
            staged.write(b"%PDF-1.4")
            source_path = staged.name
        document, job = repositories.documents.create_job(
            user_id="user-1",
            filename="sample.pdf",
            size_bytes=8,
            ttl_seconds=3600,
            source_path=source_path,
        )
        stored_job = repositories.documents.get_job(job_id=job.job_id)
        assert stored_job is not None
        stored_job.state = "processing"
        stored_job.attempts = 3
        stored_job.heartbeat_at = datetime.now(UTC) - timedelta(seconds=120)

        processed = asyncio.run(
            process_next_ingestion_job(
                repositories,
                vector_service,  # type: ignore[arg-type]
                settings=WorkerSettings(
                    poll_seconds=0.01,
                    cleanup_poll_seconds=0.01,
                    heartbeat_timeout_seconds=60,
                    max_attempts=3,
                    heartbeat_seconds=0.01,
                ),
            )
        )

        failed_job = repositories.documents.get_job(job_id=job.job_id)
        assert failed_job is not None
        self.assertTrue(processed)
        self.assertEqual(failed_job.state, "failed")
        self.assertEqual(failed_job.events[-1].payload["type"], "failed")
        self.assertIn("failed after 3 attempts", failed_job.events[-1].payload["text"])
        self.assertEqual(vector_service.deleted_documents, [("user-1", document.document_id)])
        self.assertFalse(Path(source_path).exists())


if __name__ == "__main__":
    unittest.main()
