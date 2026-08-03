import json
import unittest
from collections.abc import AsyncIterator
from unittest import TestCase
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from atlasai.service.graph_service import GraphRunner
from atlasai.web.main import create_app


class FakeGraphService(GraphRunner):
    async def run(self, _):
        return "fake assistant response"

    async def stream(self, _) -> AsyncIterator[object]:
        yield {"type": "status", "data": "[Atlas AI] LLM is working..."}
        yield {"type": "status", "data": "[Atlas AI] Calling tool: fake_tool"}
        yield {"type": "token", "data": "fake "}
        yield {"type": "token", "data": "assistant response"}
        yield {"type": "final", "data": {"messages": ["fake assistant response"]}}


def setup():
    app = create_app(FakeGraphService())
    return app


app: FastAPI = setup()

client = TestClient(app)


class TestWeb(TestCase):
    def test_home(self):
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.headers["content-type"])
        self.assertIn("Atlas AI", res.text)

    def test_health(self):
        res = client.get("/health")
        self.assertEqual((res.status_code, res.json()), (200, {"status": "ok"}))

    def test_static_assets(self):
        res = client.get("/static/styles.css")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/css", res.headers["content-type"])

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
            "atlasai.web.main.stream_ingest_pdf",
            fake_stream_ingest_pdf,
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
