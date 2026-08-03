import base64
import binascii
import mimetypes
import uuid
from pathlib import Path

IMAGE_OUTPUT_DIR = Path("tmp/extracted_images")


def build_data_url(mime_type: str, payload: bytes) -> str:
    """Encode bytes as a base64 data URL."""
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def persist_base64_image(
    image_base64: str, output_dir: Path = IMAGE_OUTPUT_DIR
) -> str | None:
    """Write a base64 image payload to disk and return the file path."""
    if not image_base64:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{uuid.uuid4()}.png"

    try:
        image_bytes = base64.b64decode(image_base64)
    except (ValueError, binascii.Error):
        return None

    image_path.write_bytes(image_bytes)
    return str(image_path)


def image_file_to_data_url(image_path: str | Path) -> str | None:
    """Read an image from disk and return a base64 data URL."""
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return None

    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "image/png"

    return build_data_url(mime_type, path.read_bytes())
