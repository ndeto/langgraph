import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import Request
from fastapi.testclient import TestClient

from atlasai.infrastructure.postgres_repositories import (
    build_in_memory_repository_bundle,
)
from atlasai.web.main import create_app
from atlasai.web.request_identity import _resolve_client_ip
from atlasai.web.routers.documents import _stream_owned_ingestion_events
from atlasai.web.streaming import stream_with_keepalive


class _FakeGraphService:
    async def stream(self, payload):
        del payload
        if False:
            yield None


class _FakeVectorService:
    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    def get_store(self, table_name: str):
        if table_name != "raggidy_docs":
            raise KeyError(table_name)
        return object()


class _SSEDocumentService:
    def __init__(self) -> None:
        self.calls = 0

    def get_owned_job(self, *, user_id: str, job_id: str):
        del user_id, job_id
        self.calls += 1
        state = "failed" if self.calls >= 3 else "processing"
        return SimpleNamespace(state=state, events=[])


def _request(*, client: str, forwarded_for: str | None = None) -> Request:
    headers = []
    if forwarded_for is not None:
        headers.append((b"x-forwarded-for", forwarded_for.encode("ascii")))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": headers,
            "client": (client, 1234),
        }
    )


class DeploymentHardeningTest(unittest.TestCase):
    def test_production_app_has_no_legacy_ingestion_route(self):
        app = create_app(
            _FakeGraphService(),  # type: ignore[arg-type]
            _FakeVectorService(),  # type: ignore[arg-type]
            build_in_memory_repository_bundle(),
        )

        paths = set(app.openapi()["paths"])

        self.assertNotIn("/ingest/pdf", paths)
        self.assertIn("/api/v1/documents", paths)

    def test_ready_reports_warmed_injected_services(self):
        app = create_app(
            _FakeGraphService(),  # type: ignore[arg-type]
            _FakeVectorService(),  # type: ignore[arg-type]
            build_in_memory_repository_bundle(),
        )

        with TestClient(app) as client:
            response = client.get("/ready")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")
        self.assertTrue(all(response.json()["checks"].values()))

    def test_ready_returns_503_before_services_are_warmed(self):
        app = create_app(
            _FakeGraphService(),  # type: ignore[arg-type]
            _FakeVectorService(),  # type: ignore[arg-type]
            build_in_memory_repository_bundle(),
        )

        response = TestClient(app).get("/ready")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "not_ready")

    def test_trusted_proxy_accepts_cidr(self):
        request = _request(client="10.0.4.12", forwarded_for="203.0.113.8")

        client_ip = _resolve_client_ip(
            request=request,
            trusted_proxies=("10.0.0.0/16",),
        )

        self.assertEqual(client_ip, "203.0.113.8")

    def test_untrusted_proxy_cannot_override_client_ip(self):
        request = _request(client="192.0.2.4", forwarded_for="203.0.113.8")

        client_ip = _resolve_client_ip(
            request=request,
            trusted_proxies=("10.0.0.0/16",),
        )

        self.assertEqual(client_ip, "192.0.2.4")

    def test_sse_stream_emits_keepalive_without_data_event(self):
        async def collect() -> list[str]:
            chunks = []
            service = _SSEDocumentService()
            with (
                patch(
                    "atlasai.web.routers.documents.SSE_KEEPALIVE_SECONDS",
                    0.0,
                ),
                patch(
                    "atlasai.web.routers.documents.SSE_POLL_SECONDS",
                    0.0,
                ),
            ):
                async for chunk in _stream_owned_ingestion_events(
                    document_service=service,  # type: ignore[arg-type]
                    user_id="user-1",
                    job_id="job-1",
                ):
                    chunks.append(chunk)
            return chunks

        chunks = asyncio.run(collect())

        self.assertIn(": keepalive\n\n", chunks)
        self.assertFalse(any(chunk.startswith("data:") for chunk in chunks))

    def test_ndjson_stream_emits_ignorable_keepalive_while_waiting(self):
        async def source():
            await asyncio.sleep(0.02)
            yield '{"type":"done"}\n'

        async def collect() -> list[str]:
            return [
                chunk
                async for chunk in stream_with_keepalive(
                    source(),
                    interval_seconds=0.005,
                )
            ]

        chunks = asyncio.run(collect())

        self.assertIn("\n", chunks)
        self.assertEqual(chunks[-1], '{"type":"done"}\n')


if __name__ == "__main__":
    unittest.main()
