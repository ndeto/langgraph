import unittest
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from atlasai.infrastructure.postgres_repositories import AssetRecord
from atlasai.infrastructure.postgres_repositories import build_in_memory_repository_bundle
from atlasai.rag.image_payloads import StoredImageAsset
from atlasai.rag.rag_ingestion import separate_content_types
from atlasai.rag.utils import build_retrieved_image_assets
from atlasai.web.dependencies import get_asset_repository
from atlasai.web.main import create_app


class FakeGraphService:
    async def stream(self, _payload) -> AsyncIterator[object]:
        if False:
            yield None


class FakeVectorService:
    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    def get_store(self, _table_name: str):
        return None


class RecordingAssetRepository:
    def __init__(self) -> None:
        self.user_id: str | None = None

    def store_asset(self, **values) -> StoredImageAsset:
        self.user_id = values["user_id"]
        return StoredImageAsset(
            asset_id="asset-1",
            mime_type=values["mime_type"],
            size_bytes=len(values["payload"]),
            checksum=values["checksum"],
        )


class Image:
    def __init__(self) -> None:
        self.metadata = SimpleNamespace(
            image_base64="aW1hZ2UtYnl0ZXM=",
            image_mime_type="image/png",
        )


class OwnedAssetRepository:
    def __init__(self) -> None:
        self.owner_id: str | None = None

    def get_owned_asset(self, *, asset_id: str, user_id: str):
        if asset_id != "asset-1" or user_id != self.owner_id:
            return None
        now = datetime.now(UTC)
        return AssetRecord(
            asset_id=asset_id,
            user_id=user_id,
            document_id="document-1",
            mime_type="image/png",
            payload=b"image-bytes",
            size_bytes=11,
            checksum="checksum",
            created_at=now,
            expires_at=now + timedelta(hours=1),
        )


class AssetsTest(unittest.TestCase):
    def test_ingestion_stores_asset_id_in_chunk_metadata(self) -> None:
        repository = RecordingAssetRepository()
        chunk = SimpleNamespace(
            text="Transformer architecture diagram",
            metadata=SimpleNamespace(orig_elements=[Image()]),
        )

        with patch(
            "atlasai.rag.rag_ingestion.summarize_image_for_retrieval",
            return_value="Transformer architecture diagram.",
        ):
            content = separate_content_types(
                chunk,
                user_id="user-1",
                document_id="document-1",
                asset_repository=repository,
            )

        self.assertEqual(repository.user_id, "user-1")
        self.assertEqual(content["images"][0]["asset_id"], "asset-1")
        assets = build_retrieved_image_assets(
            [
                SimpleNamespace(
                    metadata={
                        "original_content": '{"image_entries": [{"asset_id": "asset-1", "mime_type": "image/png"}]}'
                    }
                )
            ]
        )
        self.assertEqual(
            assets,
            [{"asset_id": "asset-1", "mime_type": "image/png"}],
        )

    def test_asset_endpoint_enforces_session_ownership(self) -> None:
        app = create_app(
            FakeGraphService(),
            FakeVectorService(),
            build_in_memory_repository_bundle(),
        )
        repository = OwnedAssetRepository()
        app.dependency_overrides[get_asset_repository] = lambda: repository
        client = TestClient(app)
        session = client.get("/api/v1/session")
        repository.owner_id = session.json()["user_id"]

        response = client.get("/api/v1/assets/asset-1")
        missing = TestClient(app).get("/api/v1/assets/missing")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"image-bytes")
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
