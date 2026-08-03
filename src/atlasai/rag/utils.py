import json
from typing import Any

from atlasai.rag.image_payloads import image_file_to_data_url


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


def extract_image_paths(chunk: Any) -> list[str]:
    metadata = getattr(chunk, "metadata", {}) or {}
    original_content = metadata.get("original_content")
    if not original_content:
        return []

    original_data = read_json_field(original_content)
    if not isinstance(original_data, dict):
        return []

    image_paths = read_json_field(original_data.get("image_paths", []))
    if isinstance(image_paths, str):
        image_paths = [image_paths] if image_paths.strip() else []

    if not isinstance(image_paths, list):
        return []

    return [str(path).strip() for path in image_paths if str(path).strip()]


def build_retrieved_image_markdown(
    chunks: list[Any],
    *,
    max_images: int = 3,
) -> str:
    image_markdown: list[str] = []
    seen_paths: set[str] = set()

    for chunk in chunks:
        for image_path in extract_image_paths(chunk):
            if image_path in seen_paths:
                continue

            seen_paths.add(image_path)
            data_url = image_file_to_data_url(image_path)
            if not data_url:
                continue

            image_index = len(image_markdown) + 1
            image_markdown.append(f"![Retrieved image {image_index}]({data_url})")
            if len(image_markdown) >= max_images:
                break

        if len(image_markdown) >= max_images:
            break

    if not image_markdown:
        return ""

    return "\n\n### Retrieved Images\n\n" + "\n\n".join(image_markdown)


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
