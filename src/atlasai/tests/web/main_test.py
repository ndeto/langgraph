import unittest
from collections.abc import AsyncIterator
from unittest import TestCase

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


if __name__ == "__main__":
    unittest.main()
