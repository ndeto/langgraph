import json
import os
import unittest
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from atlasai.application.quotas import QuotaSnapshot
from atlasai.application.usage import UsageRecord
from atlasai.infrastructure.postgres_repositories import (
    build_in_memory_repository_bundle,
)
from atlasai.service.contracts import GraphRunner
from atlasai.web.main import create_app
from atlasai.web.streaming import extract_usage_payload


class FakeVectorService:
    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    @staticmethod
    def get_store(_table_name: str):
        return None


class FakeGraphService(GraphRunner):
    async def run(self, _):
        return "fake assistant response"

    async def stream(self, _) -> AsyncIterator[object]:
        yield {"type": "status", "data": "[Atlas AI] LLM is working..."}
        yield {"type": "status", "data": "[Atlas AI] Calling tool: fake_tool"}
        yield {"type": "token", "data": "fake "}
        yield {"type": "token", "data": "assistant response"}
        yield {"type": "final", "data": {"messages": ["fake assistant response"]}}


class UsageGraphService(GraphRunner):
    async def stream(self, _) -> AsyncIterator[object]:
        yield {"type": "token", "data": "token "}
        yield {
            "type": "final",
            "data": {
                "messages": [
                    SimpleNamespace(
                        id="provider-run-1",
                        model_name="gpt-5",
                        usage_metadata={
                            "input_tokens": 12,
                            "output_tokens": 5,
                            "total_tokens": 17,
                        },
                    )
                ]
            },
        }


class ErrorGraphService(GraphRunner):
    async def stream(self, _) -> AsyncIterator[object]:
        yield {"type": "status", "data": "Starting"}
        raise RuntimeError("boom")


class CapturePayloadGraphService(GraphRunner):
    def __init__(self) -> None:
        self.payloads: list[object] = []

    async def stream(self, payload) -> AsyncIterator[object]:
        self.payloads.append(payload)
        yield {"type": "final", "data": {"messages": []}}


def setup():
    app = create_app(
        FakeGraphService(),
        FakeVectorService(),
        build_in_memory_repository_bundle(),
    )
    return app


app: FastAPI = setup()

client = TestClient(app)


class TestWeb(TestCase):
    def test_extract_usage_payload_prefers_explicit_usage_payload(self):
        payload = extract_usage_payload(
            {
                "usage_payload": {
                    "model_name": "gpt-5.4-mini-2026-03-17",
                    "usage_metadata": {
                        "input_tokens": 12,
                        "output_tokens": 5,
                        "total_tokens": 17,
                    },
                },
                "messages": [SimpleNamespace(usage_metadata=None)],
            }
        )

        self.assertEqual(
            payload,
            {
                "model_name": "gpt-5.4-mini-2026-03-17",
                "usage_metadata": {
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "total_tokens": 17,
                },
            },
        )

    def test_home(self):
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers["content-type"])
        self.assertIn("Atlas AI", res.text)

    def test_health(self):
        res = client.get("/health")
        self.assertEqual((res.status_code, res.json()), (200, {"status": "ok"}))

    def test_session_creates_cookie_and_defaults(self):
        res = client.get("/api/v1/session")

        self.assertEqual(res.status_code, 200)
        self.assertIn("set-cookie", res.headers)
        self.assertEqual(res.json()["active_document"], None)
        self.assertEqual(res.json()["active_thread"], None)
        self.assertEqual(
            res.json()["quota"],
            {
                "questions": {"limit": 10, "used": 0, "remaining": 10},
                "uploads": {"limit": 2, "used": 0, "remaining": 2},
                "tokens": {"input": 0, "output": 0, "total": 0},
            },
        )

    def test_session_restores_existing_cookie(self):
        first = client.get("/api/v1/session")
        cookie = first.cookies.get("atlas_demo_session")

        res = client.get(
            "/api/v1/session",
            cookies={"atlas_demo_session": cookie},
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["user_id"], first.json()["user_id"])

    def test_session_reads_quota_and_usage_services(self):
        app = create_app(
            FakeGraphService(),
            FakeVectorService(),
            build_in_memory_repository_bundle(),
        )
        test_client = TestClient(app)
        first_response = test_client.get("/api/v1/session")
        user_id = first_response.json()["user_id"]
        cookie_value = first_response.cookies.get("atlas_demo_session")

        app.state.repositories.quotas.set_snapshot(
            user_id=user_id,
            snapshot=QuotaSnapshot(questions_used=4, uploads_used=1),
        )
        app.state.repositories.usage.append_record(
            user_id=user_id,
            record=UsageRecord(
                operation="agent",
                input_tokens=120,
                output_tokens=30,
                total_tokens=150,
            ),
        )

        res = test_client.get(
            "/api/v1/session",
            cookies={"atlas_demo_session": cookie_value},
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(
            res.json()["quota"],
            {
                "questions": {"limit": 10, "used": 4, "remaining": 6},
                "uploads": {"limit": 2, "used": 1, "remaining": 1},
                "tokens": {"input": 120, "output": 30, "total": 150},
            },
        )

    def test_invoke_rejects_when_question_quota_is_exhausted(self):
        app = create_app(
            FakeGraphService(),
            FakeVectorService(),
            build_in_memory_repository_bundle(),
        )
        test_client = TestClient(app)
        first_response = test_client.get("/api/v1/session")
        user_id = first_response.json()["user_id"]

        app.state.repositories.quotas.set_snapshot(
            user_id=user_id,
            snapshot=QuotaSnapshot(questions_used=10),
        )

        res = test_client.post(
            "/invoke",
            json={"user_input": "Hello", "thread_id": "test-thread"},
        )

        self.assertEqual(res.status_code, 429)
        self.assertEqual(res.json(), {"detail": "Question quota exceeded."})
        self.assertIn("x-trace-id", res.headers)

    def test_ingest_pdf_rejects_when_upload_quota_is_exhausted(self):
        app = create_app(
            FakeGraphService(),
            FakeVectorService(),
            build_in_memory_repository_bundle(),
        )
        test_client = TestClient(app)
        first_response = test_client.get("/api/v1/session")
        user_id = first_response.json()["user_id"]

        app.state.repositories.quotas.set_snapshot(
            user_id=user_id,
            snapshot=QuotaSnapshot(uploads_used=2),
        )

        res = test_client.post(
            "/ingest/pdf",
            files={"file": ("sample.pdf", b"%PDF-1.4", "application/pdf")},
        )

        self.assertEqual(res.status_code, 429)
        self.assertEqual(res.json(), {"detail": "Upload quota exceeded."})
        self.assertIn("x-trace-id", res.headers)

    def test_invoke_rejects_when_request_key_question_quota_is_exhausted(self):
        with patch.dict(os.environ, {"ATLAS_DEMO_IP_QUESTION_LIMIT": "1"}):
            app = create_app(
                FakeGraphService(),
                FakeVectorService(),
                build_in_memory_repository_bundle(),
            )

        first_client = TestClient(app)
        second_client = TestClient(app)

        first_client.get("/api/v1/session")
        first = first_client.post(
            "/invoke",
            headers={"x-atlas-request-key": "hash-1"},
            json={"user_input": "Hello", "thread_id": "thread-1"},
        )

        second_client.get("/api/v1/session")
        second = second_client.post(
            "/invoke",
            headers={"x-atlas-request-key": "hash-1"},
            json={"user_input": "Hello again", "thread_id": "thread-2"},
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertEqual(
            second.json(),
            {"detail": "Question quota exceeded for this request key."},
        )

    def test_ingest_pdf_rejects_when_request_key_upload_quota_is_exhausted(self):
        with patch.dict(os.environ, {"ATLAS_DEMO_IP_UPLOAD_LIMIT": "1"}):
            app = create_app(
                FakeGraphService(),
                FakeVectorService(),
                build_in_memory_repository_bundle(),
            )

        first_client = TestClient(app)
        second_client = TestClient(app)

        async def fake_stream_ingest_pdf(*_, **__):
            yield {"type": "done", "text": "Ingestion complete"}

        with patch(
            "atlasai.web.main.get_stream_ingest_pdf",
            return_value=fake_stream_ingest_pdf,
        ):
            first_client.get("/api/v1/session")
            first = first_client.post(
                "/ingest/pdf",
                headers={"x-atlas-request-key": "hash-1"},
                files={"file": ("sample.pdf", b"%PDF-1.4", "application/pdf")},
            )

            second_client.get("/api/v1/session")
            second = second_client.post(
                "/ingest/pdf",
                headers={"x-atlas-request-key": "hash-1"},
                files={"file": ("sample.pdf", b"%PDF-1.4", "application/pdf")},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertEqual(
            second.json(),
            {"detail": "Upload quota exceeded for this request key."},
        )

    def test_static_assets(self):
        index = client.get("/")
        self.assertEqual(index.status_code, 200)
        self.assertIn("/static/assets/", index.text)

    def test_invoke(self):
        res = client.post(
            "invoke",
            json={
                "user_input": "What does KDuka do?",
                "thread_id": "test-thread",
            },
        )

        self.assertEqual(res.status_code, 200)
        self.assertIn("text/plain", res.headers["content-type"])
        self.assertEqual(
            res.text,
            "[Atlas AI] LLM is working...\n"
            "[Atlas AI] Calling tool: fake_tool\n"
            "fake assistant response",
        )

    def test_invoke_ndjson(self):
        res = client.post(
            "invoke?stream_format=ndjson",
            json={"user_input": "Hello", "thread_id": "test-thread"},
        )

        events = [json.loads(line) for line in res.text.splitlines()]

        self.assertEqual(res.status_code, 200)
        self.assertIn("application/x-ndjson", res.headers["content-type"])
        self.assertEqual(
            events,
            [
                {"type": "status", "text": "[Atlas AI] LLM is working..."},
                {"type": "status", "text": "[Atlas AI] Calling tool: fake_tool"},
                {"type": "token", "text": "fake "},
                {"type": "token", "text": "assistant response"},
                {"type": "done"},
            ],
        )

    def test_invoke_injects_session_user_id_into_graph_payload(self):
        graph_service = CapturePayloadGraphService()
        app = create_app(
            graph_service,
            FakeVectorService(),
            build_in_memory_repository_bundle(),
        )
        test_client = TestClient(app)
        session_response = test_client.get("/api/v1/session")
        cookie_value = session_response.cookies.get("atlas_demo_session")
        user_id = session_response.json()["user_id"]

        res = test_client.post(
            "/invoke",
            cookies={"atlas_demo_session": cookie_value},
            json={"user_input": "Hello", "thread_id": "thread-1", "user_id": "ignored"},
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(
            graph_service.payloads,
            [{"user_input": "Hello", "thread_id": "thread-1", "user_id": user_id}],
        )

    def test_invoke_records_provider_usage_once(self):
        app = create_app(
            UsageGraphService(),
            FakeVectorService(),
            build_in_memory_repository_bundle(),
        )
        test_client = TestClient(app)
        session_response = test_client.get("/api/v1/session")
        cookie_value = session_response.cookies.get("atlas_demo_session")

        first = test_client.post(
            "/invoke",
            headers={"x-trace-id": "trace-fixed"},
            cookies={"atlas_demo_session": cookie_value},
            json={"user_input": "Hello", "thread_id": "thread-1"},
        )
        second = test_client.post(
            "/invoke",
            headers={"x-trace-id": "trace-fixed"},
            cookies={"atlas_demo_session": cookie_value},
            json={"user_input": "Hello", "thread_id": "thread-1"},
        )
        session_after = test_client.get(
            "/api/v1/session",
            cookies={"atlas_demo_session": cookie_value},
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(
            session_after.json()["quota"]["tokens"],
            {"input": 12, "output": 5, "total": 17},
        )

    def test_create_thread_sets_active_thread_cookie(self):
        app = create_app(
            FakeGraphService(),
            FakeVectorService(),
            build_in_memory_repository_bundle(),
        )
        test_client = TestClient(app)
        session_response = test_client.get("/api/v1/session")
        cookie_value = session_response.cookies.get("atlas_demo_session")

        res = test_client.post(
            "/api/v1/threads",
            cookies={"atlas_demo_session": cookie_value},
        )
        session_after = test_client.get(
            "/api/v1/session",
            cookies={"atlas_demo_session": res.cookies.get("atlas_demo_session")},
        )

        self.assertEqual(res.status_code, 201)
        self.assertIn("thread_id", res.json())
        self.assertEqual(
            session_after.json()["active_thread"],
            res.json()["thread_id"],
        )

    def test_get_thread_rejects_other_users(self):
        app = create_app(
            FakeGraphService(),
            FakeVectorService(),
            build_in_memory_repository_bundle(),
        )
        first_client = TestClient(app)
        second_client = TestClient(app)

        first_session = first_client.get("/api/v1/session")
        thread = first_client.post(
            "/api/v1/threads",
            cookies={
                "atlas_demo_session": first_session.cookies.get("atlas_demo_session")
            },
        )

        second_session = second_client.get("/api/v1/session")
        res = second_client.get(
            f"/api/v1/threads/{thread.json()['thread_id']}",
            cookies={
                "atlas_demo_session": second_session.cookies.get("atlas_demo_session")
            },
        )

        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.json(), {"detail": "Thread not found."})

    def test_thread_messages_stream_structured_ndjson(self):
        app = create_app(
            UsageGraphService(),
            FakeVectorService(),
            build_in_memory_repository_bundle(),
        )
        test_client = TestClient(app)
        session_response = test_client.get("/api/v1/session")
        cookie_value = session_response.cookies.get("atlas_demo_session")
        thread_response = test_client.post(
            "/api/v1/threads",
            cookies={"atlas_demo_session": cookie_value},
        )

        res = test_client.post(
            f"/api/v1/threads/{thread_response.json()['thread_id']}/messages",
            headers={"x-trace-id": "trace-thread"},
            cookies={"atlas_demo_session": thread_response.cookies.get("atlas_demo_session")},
            json={"user_input": "Hello"},
        )

        events = [json.loads(line) for line in res.text.splitlines()]

        self.assertEqual(res.status_code, 200)
        self.assertIn("application/x-ndjson", res.headers["content-type"])
        self.assertEqual(
            events,
            [
                {"type": "token", "text": "token "},
                {
                    "type": "usage",
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "total_tokens": 17,
                    "status": "known",
                },
                {
                    "type": "done",
                    "thread_id": thread_response.json()["thread_id"],
                },
            ],
        )

    def test_thread_messages_emit_error_event(self):
        app = create_app(
            ErrorGraphService(),
            FakeVectorService(),
            build_in_memory_repository_bundle(),
        )
        test_client = TestClient(app)
        session_response = test_client.get("/api/v1/session")
        thread_response = test_client.post(
            "/api/v1/threads",
            cookies={
                "atlas_demo_session": session_response.cookies.get(
                    "atlas_demo_session"
                )
            },
        )

        res = test_client.post(
            f"/api/v1/threads/{thread_response.json()['thread_id']}/messages",
            cookies={"atlas_demo_session": thread_response.cookies.get("atlas_demo_session")},
            json={"user_input": "Hello"},
        )

        events = [json.loads(line) for line in res.text.splitlines()]

        self.assertEqual(res.status_code, 200)
        self.assertEqual(
            events,
            [
                {"type": "status", "text": "Starting"},
                {
                    "type": "usage",
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "status": "unknown",
                },
                {"type": "error", "text": "Request failed."},
            ],
        )

    def test_documents_api_accepts_upload_and_exposes_events(self):
        app = create_app(
            FakeGraphService(),
            FakeVectorService(),
            build_in_memory_repository_bundle(),
        )
        test_client = TestClient(app)
        session_response = test_client.get("/api/v1/session")
        cookie_value = session_response.cookies.get("atlas_demo_session")

        async def fake_stream_ingest_pdf(*_, **__):
            yield {"type": "file", "file_name": "sample.pdf"}
            yield {"type": "log", "text": "Preparing ingestion for sample.pdf"}
            yield {"type": "done", "text": "Ingestion complete"}

        with patch(
            "atlasai.web.routers.documents.get_stream_ingest_pdf",
            return_value=fake_stream_ingest_pdf,
        ):
            accepted = test_client.post(
                "/api/v1/documents",
                cookies={"atlas_demo_session": cookie_value},
                files={"file": ("sample.pdf", b"%PDF-1.4", "application/pdf")},
            )
            events_response = test_client.get(
                f"/api/v1/ingestions/{accepted.json()['job_id']}/events",
                cookies={"atlas_demo_session": cookie_value},
            )
            session_after = test_client.get(
                "/api/v1/session",
                cookies={"atlas_demo_session": cookie_value},
            )

        self.assertEqual(accepted.status_code, 202)
        self.assertIn("document_id", accepted.json())
        self.assertIn("job_id", accepted.json())
        self.assertIn("text/event-stream", events_response.headers["content-type"])
        self.assertIn('"type": "queued"', events_response.text)
        self.assertIn('"type": "ready"', events_response.text)
        self.assertEqual(
            session_after.json()["active_document"],
            accepted.json()["document_id"],
        )

    def test_documents_events_reject_other_users(self):
        app = create_app(
            FakeGraphService(),
            FakeVectorService(),
            build_in_memory_repository_bundle(),
        )
        first_client = TestClient(app)
        second_client = TestClient(app)

        async def fake_stream_ingest_pdf(*_, **__):
            yield {"type": "done", "text": "Ingestion complete"}

        first_session = first_client.get("/api/v1/session")
        with patch(
            "atlasai.web.routers.documents.get_stream_ingest_pdf",
            return_value=fake_stream_ingest_pdf,
        ):
            accepted = first_client.post(
                "/api/v1/documents",
                cookies={
                    "atlas_demo_session": first_session.cookies.get("atlas_demo_session")
                },
                files={"file": ("sample.pdf", b"%PDF-1.4", "application/pdf")},
            )

        second_session = second_client.get("/api/v1/session")
        res = second_client.get(
            f"/api/v1/ingestions/{accepted.json()['job_id']}/events",
            cookies={
                "atlas_demo_session": second_session.cookies.get("atlas_demo_session")
            },
        )

        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.json(), {"detail": "Ingestion job not found."})

    def test_ingest_pdf_records_unknown_usage_without_affecting_token_totals(self):
        app = create_app(
            FakeGraphService(),
            FakeVectorService(),
            build_in_memory_repository_bundle(),
        )
        test_client = TestClient(app)
        session_response = test_client.get("/api/v1/session")
        cookie_value = session_response.cookies.get("atlas_demo_session")
        user_id = session_response.json()["user_id"]

        async def fake_stream_ingest_pdf(*_, **__):
            yield {"type": "done", "text": "Ingestion complete"}

        with patch(
            "atlasai.web.main.get_stream_ingest_pdf",
            return_value=fake_stream_ingest_pdf,
        ):
            res = test_client.post(
                "/ingest/pdf",
                headers={"x-trace-id": "trace-ingest"},
                cookies={"atlas_demo_session": cookie_value},
                files={"file": ("sample.pdf", b"%PDF-1.4", "application/pdf")},
            )

        records = app.state.repositories.usage.list_records(user_id=user_id)
        session_after = test_client.get(
            "/api/v1/session",
            cookies={"atlas_demo_session": cookie_value},
        )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].operation, "ingestion")
        self.assertEqual(records[0].status, "unknown")
        self.assertEqual(
            session_after.json()["quota"]["tokens"],
            {"input": 0, "output": 0, "total": 0},
        )

    def test_ingest_pdf_rejects_non_pdf_uploads(self):
        res = client.post(
            "/ingest/pdf",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )

        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json(), {"detail": "Only PDF uploads are supported."})

    def test_ingest_pdf_streams_ingestion_events(self):
        async def fake_stream_ingest_pdf(*_, **__):
            yield {"type": "file", "file_name": "sample.pdf"}
            yield {"type": "log", "text": "Preparing ingestion for sample.pdf"}
            yield {"type": "stats", "elements": 8, "chunks": 3, "docs": 2}
            yield {
                "type": "done",
                "text": "Ingestion complete",
                "file_name": "sample.pdf",
                "elements": 8,
                "chunks": 3,
                "docs": 2,
            }

        with patch(
            "atlasai.web.main.get_stream_ingest_pdf",
            return_value=fake_stream_ingest_pdf,
        ):
            res = client.post(
                "/ingest/pdf?stream_format=ndjson",
                files={"file": ("sample.pdf", b"%PDF-1.4", "application/pdf")},
            )

        events = [json.loads(line) for line in res.text.splitlines()]

        self.assertEqual(res.status_code, 200)
        self.assertIn("application/x-ndjson", res.headers["content-type"])
        self.assertEqual(
            events,
            [
                {"type": "file", "file_name": "sample.pdf"},
                {"type": "log", "text": "Preparing ingestion for sample.pdf"},
                {"type": "stats", "elements": 8, "chunks": 3, "docs": 2},
                {
                    "type": "done",
                    "text": "Ingestion complete",
                    "file_name": "sample.pdf",
                    "elements": 8,
                    "chunks": 3,
                    "docs": 2,
                },
            ],
        )


if __name__ == "__main__":
    unittest.main()
