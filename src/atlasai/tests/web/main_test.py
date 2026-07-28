import unittest
from unittest import TestCase

from fastapi import FastAPI
from fastapi.testclient import TestClient

from atlasai.service.graph import GraphRunner
from atlasai.web.main import create_app


class FakeGraphService(GraphRunner):
    def run(self, _):
        return "fake assistant response"


def setup():
    app = create_app(FakeGraphService())
    return app


app: FastAPI = setup()

client = TestClient(app)


class TestWeb(TestCase):
    def test_home(self):
        res = client.get("/")
        self.assertEqual(res.status_code, 200)

    def test_invoke(self):
        res = client.post(
            "invoke",
            json={
                "user_input": "What does KDuka do?",
                "thread_id": "test-thread",
            },
        )

        self.assertEqual((res.status_code, res.text), (200, "fake assistant response"))


if __name__ == "__main__":
    unittest.main()
