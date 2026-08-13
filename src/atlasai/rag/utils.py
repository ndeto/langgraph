import json
from typing import Any


def read_json_field(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def doc_to_prompt_text(chunk: Any) -> str:
    metadata = getattr(chunk, "metadata", {}) or {}
    original_content = metadata.get("original_content")

    if original_content:
        original_data = read_json_field(original_content)
        if not isinstance(original_data, dict):
            original_data = {}

        raw_text = read_json_field(original_data.get("raw_text", ""))
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)

        tables_html = read_json_field(original_data.get("tables_html", []))
        if isinstance(tables_html, str):
            tables_html = [tables_html] if tables_html.strip() else []

        parts: list[str] = []
        if raw_text.strip():
            parts.append(raw_text.strip())

        if tables_html:
            table_text = "\n\n".join(
                str(table).strip() for table in tables_html if str(table).strip()
            )
            if table_text:
                parts.append(table_text)

        return "\n\n".join(parts).strip()

    page_content = getattr(chunk, "page_content", "")
    if not isinstance(page_content, str):
        page_content = str(page_content)

    return page_content.strip()


def extract_image_entries(chunk: Any) -> list[dict[str, Any]]:
    metadata = getattr(chunk, "metadata", {}) or {}
    original_content = metadata.get("original_content")
    if not original_content:
        return []

    original_data = read_json_field(original_content)
    if not isinstance(original_data, dict):
        return []

    image_entries = read_json_field(original_data.get("image_entries", []))
    if not isinstance(image_entries, list):
        return []

    return [entry for entry in image_entries if isinstance(entry, dict)]


def build_retrieved_image_assets(
    chunks: list[Any],
    *,
    max_images: int = 3,
) -> list[dict[str, str]]:
    """Return images from retrieved chunks in the top-ranked document."""

    assets: list[dict[str, str]] = []
    seen_asset_ids: set[str] = set()
    primary_document_id = None
    if chunks:
        first_metadata = getattr(chunks[0], "metadata", {}) or {}
        primary_document_id = first_metadata.get("document_id")

    for chunk in chunks:
        metadata = getattr(chunk, "metadata", {}) or {}
        if (
            primary_document_id is not None
            and metadata.get("document_id") != primary_document_id
        ):
            continue

        for image_entry in extract_image_entries(chunk):
            asset_id = str(image_entry.get("asset_id", "")).strip()
            if not asset_id or asset_id in seen_asset_ids:
                continue

            seen_asset_ids.add(asset_id)
            mime_type = str(image_entry.get("mime_type", "image/png")).strip()
            assets.append(
                {
                    "asset_id": asset_id,
                    "mime_type": mime_type or "image/png",
                }
            )
            if len(assets) >= max_images:
                return assets

        if len(assets) >= max_images:
            break

    return assets


def build_document_context_prompt(
    query: str, chunks: list[Any], source_label: str
) -> str:

    prompt_text = f"""Based on the following {source_label} documents, answer this question: {query}

    CONTENT TO ANALYZE:
    """
    document_count = 0

    for chunk in chunks:
        chunk_text = doc_to_prompt_text(chunk)
        if not chunk_text:
            continue

        document_count += 1
        prompt_text += f"--- Document {document_count} ---\n"
        prompt_text += f"{chunk_text}\n\n"

    if document_count == 0:
        prompt_text += "No non-empty documents were retrieved for this source.\n\n"

    prompt_text += """Use the document content above when it is relevant.
If the documents do not contain enough information, say so instead of guessing."""

    return prompt_text
