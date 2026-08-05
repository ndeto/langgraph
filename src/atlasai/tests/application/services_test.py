import unittest
from types import SimpleNamespace

from atlasai.application.quotas import (
    AdmissionResult,
    BucketQuotaSnapshot,
    InMemoryQuotaRepository,
    QuotaPolicy,
    QuotaService,
)
from atlasai.application.usage import (
    InMemoryUsageRepository,
    UsageRecord,
    UsageService,
)
from atlasai.application.threads import InMemoryThreadRepository, ThreadService
from atlasai.application.sessions import SessionCookieSettings, SessionService
from atlasai.domain.models import TokenUsageSummary
from atlasai.infrastructure.telemetry import (
    TelemetryContext,
    build_telemetry_metadata,
    ensure_trace_id,
    usage_record_from_callback,
)
from atlasai.rag.rag_ingestion import summarize_chunk


class TestUsageService(unittest.TestCase):
    def test_usage_summary_aggregates_known_records(self):
        repository = InMemoryUsageRepository()
        service = UsageService(repository=repository)
        repository.append_record(
            user_id="user-1",
            record=UsageRecord(
                operation="agent",
                input_tokens=120,
                output_tokens=30,
                total_tokens=150,
            ),
        )
        repository.append_record(
            user_id="user-1",
            record=UsageRecord(
                operation="embedding",
                input_tokens=80,
                output_tokens=0,
                total_tokens=80,
            ),
        )

        summary = service.get_token_summary(user_id="user-1")

        self.assertEqual(
            summary,
            TokenUsageSummary(input=200, output=30, total=230),
        )

    def test_usage_summary_skips_unknown_records(self):
        repository = InMemoryUsageRepository()
        service = UsageService(repository=repository)
        repository.append_record(
            user_id="user-1",
            record=UsageRecord(
                operation="agent",
                input_tokens=100,
                output_tokens=20,
                total_tokens=120,
                status="unknown",
            ),
        )

        summary = service.get_token_summary(user_id="user-1")

        self.assertEqual(summary, TokenUsageSummary())

    def test_record_usage_dedupes_by_run_id(self):
        repository = InMemoryUsageRepository()
        service = UsageService(repository=repository)

        first = service.record_usage(
            user_id="user-1",
            record=UsageRecord(
                operation="agent",
                run_id="run-1",
                input_tokens=120,
                output_tokens=30,
                total_tokens=150,
            ),
        )
        second = service.record_usage(
            user_id="user-1",
            record=UsageRecord(
                operation="agent",
                run_id="run-1",
                input_tokens=120,
                output_tokens=30,
                total_tokens=150,
            ),
        )

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(len(repository.list_records(user_id="user-1")), 1)


class TestRagIngestion(unittest.TestCase):
    def test_summarize_chunk_stores_user_id_in_metadata(self):
        chunk = SimpleNamespace(
            text="Atlas AI document text",
            metadata=SimpleNamespace(orig_elements=[]),
        )

        document = summarize_chunk(
            chunk,
            current_chunk=1,
            total_chunks=1,
            user_id="user-123",
        )

        self.assertEqual(document.metadata["user_id"], "user-123")

    def test_summarize_chunk_stores_document_id_in_metadata(self):
        chunk = SimpleNamespace(
            text="Atlas AI document text",
            metadata=SimpleNamespace(orig_elements=[]),
        )

        document = summarize_chunk(
            chunk,
            current_chunk=1,
            total_chunks=1,
            document_id="document-123",
        )

        self.assertEqual(document.metadata["document_id"], "document-123")


class TestQuotaService(unittest.TestCase):
    def test_quota_summary_uses_snapshot_and_policy(self):
        repository = InMemoryQuotaRepository()
        repository.set_client_snapshot(
            client_key="browser-1",
            snapshot=BucketQuotaSnapshot(
                questions_used=3,
                uploads_used=1,
            ),
        )
        service = QuotaService(
            policy=QuotaPolicy(
                user_question_limit=10,
                user_upload_limit=2,
                ip_question_limit=30,
                ip_upload_limit=4,
                concurrent_ingestions_per_user=1,
                concurrent_agent_runs_per_user=1,
            ),
            repository=repository,
            client_key="browser-1",
            ip_hash="network-1",
        )

        summary = service.get_session_quota_summary(
            user_id="user-1",
            token_usage=TokenUsageSummary(input=50, output=10, total=60),
        )

        self.assertEqual(summary.questions.limit, 10)
        self.assertEqual(summary.questions.used, 3)
        self.assertEqual(summary.questions.remaining, 7)
        self.assertEqual(summary.uploads.limit, 2)
        self.assertEqual(summary.uploads.used, 1)
        self.assertEqual(summary.uploads.remaining, 1)
        self.assertEqual(summary.tokens.total, 60)

    def test_claim_question_enforces_limit_and_release(self):
        repository = InMemoryQuotaRepository()
        service = QuotaService(
            policy=QuotaPolicy(
                user_question_limit=1,
                user_upload_limit=2,
                ip_question_limit=30,
                ip_upload_limit=4,
                concurrent_ingestions_per_user=1,
                concurrent_agent_runs_per_user=1,
            ),
            repository=repository,
            client_key="browser-1",
            ip_hash="network-1",
        )

        first = service.claim_question(user_id="user-1")
        second = service.claim_question(user_id="user-1")

        self.assertEqual(first, AdmissionResult(allowed=True, code=None, message=None))
        self.assertEqual(
            second,
            AdmissionResult(
                allowed=False,
                code="browser_question_quota_reached",
                message="Question quota reached for this browser session.",
            ),
        )

        service.release_agent_run(user_id="user-1")
        snapshot = repository.get_snapshot(user_id="user-1")
        self.assertEqual(snapshot.active_agent_runs, 0)

    def test_claim_upload_enforces_concurrency(self):
        repository = InMemoryQuotaRepository()
        service = QuotaService(
            policy=QuotaPolicy(
                user_question_limit=10,
                user_upload_limit=2,
                ip_question_limit=30,
                ip_upload_limit=4,
                concurrent_ingestions_per_user=1,
                concurrent_agent_runs_per_user=1,
            ),
            repository=repository,
            client_key="browser-1",
            ip_hash="network-1",
        )

        first = service.claim_upload(user_id="user-1")
        second = service.claim_upload(user_id="user-1")

        self.assertEqual(first, AdmissionResult(allowed=True, code=None, message=None))
        self.assertEqual(
            second,
            AdmissionResult(
                allowed=False,
                code="ingestion_in_progress",
                message="Another document is still processing.",
            ),
        )

    def test_claim_question_enforces_request_key_limit_across_users(self):
        repository = InMemoryQuotaRepository()
        service = QuotaService(
            policy=QuotaPolicy(
                user_question_limit=10,
                user_upload_limit=2,
                ip_question_limit=1,
                ip_upload_limit=4,
                concurrent_ingestions_per_user=1,
                concurrent_agent_runs_per_user=1,
            ),
            repository=repository,
        )

        first = service.claim_question(
            user_id="user-1",
            client_key="browser-1",
            ip_hash="hash-1",
        )
        second = service.claim_question(
            user_id="user-2",
            client_key="browser-2",
            ip_hash="hash-1",
        )

        self.assertEqual(first, AdmissionResult(allowed=True, code=None, message=None))
        self.assertEqual(
            second,
            AdmissionResult(
                allowed=False,
                code="network_question_quota_reached",
                message="Question quota reached for this network.",
            ),
        )
        self.assertEqual(
            repository.get_ip_snapshot(ip_hash="hash-1"),
            BucketQuotaSnapshot(questions_used=1, uploads_used=0),
        )

    def test_claim_upload_enforces_request_key_limit_across_users(self):
        repository = InMemoryQuotaRepository()
        service = QuotaService(
            policy=QuotaPolicy(
                user_question_limit=10,
                user_upload_limit=2,
                ip_question_limit=30,
                ip_upload_limit=1,
                concurrent_ingestions_per_user=1,
                concurrent_agent_runs_per_user=1,
            ),
            repository=repository,
        )

        first = service.claim_upload(
            user_id="user-1",
            client_key="browser-1",
            ip_hash="hash-1",
        )
        second = service.claim_upload(
            user_id="user-2",
            client_key="browser-2",
            ip_hash="hash-1",
        )

        self.assertEqual(first, AdmissionResult(allowed=True, code=None, message=None))
        self.assertEqual(
            second,
            AdmissionResult(
                allowed=False,
                code="network_upload_quota_reached",
                message="Upload quota reached for this network.",
            ),
        )
        self.assertEqual(
            repository.get_ip_snapshot(ip_hash="hash-1"),
            BucketQuotaSnapshot(questions_used=0, uploads_used=1),
        )


class TestThreadService(unittest.TestCase):
    def test_thread_service_returns_owned_thread_only(self):
        repository = InMemoryThreadRepository()
        service = ThreadService(repository=repository)

        thread = service.create_thread(
            user_id="user-1",
            expires_at=SessionService(
                SessionCookieSettings(
                    name="atlas_demo_session",
                    secret="secret",
                    max_age_seconds=60,
                    secure=False,
                )
            ).resolve_session(None).session.expires_at,
        )

        self.assertIsNotNone(
            service.get_owned_thread(user_id="user-1", thread_id=thread.thread_id)
        )
        self.assertIsNone(
            service.get_owned_thread(user_id="user-2", thread_id=thread.thread_id)
        )


class TestTelemetryMetadata(unittest.TestCase):
    def test_metadata_omits_none_values(self):
        metadata = build_telemetry_metadata(
            TelemetryContext(
                operation="agent",
                user_id="user-1",
                trace_id="trace-1",
            )
        )

        self.assertEqual(
            metadata,
            {
                "operation": "agent",
                "user_id": "user-1",
                "trace_id": "trace-1",
            },
        )

    def test_ensure_trace_id_returns_existing_or_new_value(self):
        self.assertEqual(ensure_trace_id("trace-123"), "trace-123")
        self.assertTrue(ensure_trace_id(None))

    def test_usage_record_from_callback_reads_usage_metadata(self):
        record = usage_record_from_callback(
            operation="agent",
            payload={
                "run_id": "run-1",
                "provider": "openai",
                "model_name": "gpt-5",
                "usage_metadata": {
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "total_tokens": 150,
                },
            },
            trace_id="trace-1",
        )

        self.assertEqual(
            record,
            UsageRecord(
                operation="agent",
                run_id="run-1",
                provider="openai",
                model="gpt-5",
                input_tokens=120,
                output_tokens=30,
                total_tokens=150,
                status="known",
                trace_id="trace-1",
            ),
        )

    def test_usage_record_from_callback_reads_response_token_usage(self):
        record = usage_record_from_callback(
            operation="agent",
            payload={
                "response_metadata": {
                    "id": "resp-1",
                    "model_name": "gpt-5-mini",
                    "token_usage": {
                        "prompt_tokens": 80,
                        "completion_tokens": 20,
                        "total_tokens": 100,
                    },
                }
            },
            provider="openai",
        )

        self.assertEqual(record.run_id, "resp-1")
        self.assertEqual(record.model, "gpt-5-mini")
        self.assertEqual(record.input_tokens, 80)
        self.assertEqual(record.output_tokens, 20)
        self.assertEqual(record.total_tokens, 100)
        self.assertEqual(record.status, "known")

    def test_usage_record_from_callback_marks_missing_usage_unknown(self):
        record = usage_record_from_callback(
            operation="embedding",
            payload={"run_id": "run-2", "provider": "openai"},
        )

        self.assertEqual(
            record,
            UsageRecord(
                operation="embedding",
                run_id="run-2",
                provider="openai",
                status="unknown",
            ),
        )


if __name__ == "__main__":
    unittest.main()
