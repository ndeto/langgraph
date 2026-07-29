import base64
import binascii
import uuid
from pathlib import Path

IMAGE_OUTPUT_DIR = Path("tmp/extracted_images")


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
