import base64
import binascii
from dataclasses import dataclass
from typing import Protocol

MAX_ASSETS_PER_DOCUMENT = 20
MAX_ASSET_SIZE_BYTES = 5 * 1024 * 1024
MAX_DOCUMENT_ASSET_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class StoredImageAsset:
    """Persisted image metadata safe for vector storage."""

    asset_id: str
    mime_type: str
    size_bytes: int
    checksum: str


class ImageAssetRepository(Protocol):
    """Persistence boundary for extracted document images."""

    def store_asset(
        self,
        *,
        user_id: str,
        document_id: str,
        mime_type: str,
        payload: bytes,
        checksum: str,
    ) -> StoredImageAsset: ...


def build_data_url(mime_type: str, payload: bytes) -> str:
    """Encode bytes as a base64 data URL."""

    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def decode_base64_image(image_base64: str | None) -> bytes | None:
    """Decode an extracted image payload without writing it to disk."""

    if not image_base64:
        return None
    encoded = image_base64.split(",", 1)[-1]
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        return None
