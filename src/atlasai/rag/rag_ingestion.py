import asyncio
import hashlib
import json
import os
import platform
import re
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVectorStore
from langchain_unstructured.document_loaders import Element
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from atlasai.config.sys_config import SysConfig, bootstrap_config
from atlasai.rag.image_payloads import (
    ImageAssetRepository,
    StoredImageAsset,
    build_data_url,
    decode_base64_image,
)
from atlasai.store.hybrid_store import PostgresVectorService

config: SysConfig = bootstrap_config()

load_dotenv()


class IngestionEvent(TypedDict, total=False):
    type: str
    text: str
    file_name: str
    elements: int
    chunks: int
    docs: int
    current: int
    total: int


def build_ingestion_event(
    event_type: str,
    text: str | None = None,
    **extra: Any,
) -> IngestionEvent:
    event = cast(IngestionEvent, {"type": event_type, **extra})
    if text is not None:
        event["text"] = text
    return event


def emit_log(logger: Callable[[IngestionEvent], None] | None, text: str) -> None:
    print(text)
    if logger is not None:
        logger(build_ingestion_event("log", text=text))


def emit_stats(
    logger: Callable[[IngestionEvent], None] | None,
    *,
    file_name: str,
    elements: int,
    chunks: int,
    docs: int,
) -> None:
    if logger is not None:
        logger(
            build_ingestion_event(
                "stats",
                file_name=file_name,
                elements=elements,
                chunks=chunks,
                docs=docs,
            )
        )


def partition_pdf_doc(
    path: str | Path,
    logger: Callable[[IngestionEvent], None] | None = None,
):
    """Partitions the document"""
    resolved_path = Path(path)
    sample_bytes: bytes | None = None
    read_error: str | None = None
    try:
        with resolved_path.open("rb") as pdf_file:
            sample_bytes = pdf_file.read(32)
    except Exception as exc:
        read_error = f"{type(exc).__name__}: {exc}"
    emit_log(
        logger,
        (
            "Partitioning document "
            f"(path={resolved_path}, exists={resolved_path.exists()}, "
            f"size_bytes={resolved_path.stat().st_size if resolved_path.exists() else 'missing'}, "
            f"cwd={Path.cwd()}, platform={platform.platform()}, "
            f"pid={os.getpid()}, readable={read_error is None}, "
            f"read_error={read_error}, sample_header={sample_bytes!r}, "
            "strategy=hi_res, extract_images=True)"
        ),
    )
    try:
        elements = partition_pdf(
            filename=str(resolved_path),
            strategy="hi_res",
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
        )
    except Exception as exc:
        emit_log(
            logger,
            f"Partitioning failed in hi_res mode: {type(exc).__name__}: {exc}",
        )
        raise

    emit_log(logger, f"Found {len(elements)} elements")
    return elements


def create_chunks_by_title(
    elements: list[Element],
    logger: Callable[[IngestionEvent], None] | None = None,
):
    """Create Intelligent chunks"""
    emit_log(logger, "Creating smart chunks")
    chunks = chunk_by_title(
        elements=elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500,
    )

    emit_log(logger, f"Created {len(chunks)} chunks")
    return chunks


class ContentData(TypedDict):
    text: str | None
    tables: list
    images: list
    types: list


class ImageEntry(TypedDict, total=False):
    asset_id: str
    mime_type: str
    checksum: str
    context_text: str
    summary: str


def extract_chunk_text(chunk: Element) -> str:
    """Extract usable text from HTML or PDF chunks."""
    text = getattr(chunk, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    page_content = getattr(chunk, "page_content", None)
    if isinstance(page_content, str) and page_content.strip():
        return page_content.strip()

    return str(chunk).strip()


def normalize_image_context(context_text: str, *, max_chars: int = 220) -> str:
    """Compress nearby chunk text into a short retrieval hint."""

    compact = re.sub(r"\s+", " ", context_text).strip()
    if not compact:
        return ""

    sentences = re.split(r"(?<=[.!?])\s+", compact)
    excerpt = " ".join(sentence.strip() for sentence in sentences[:2] if sentence.strip())
    if not excerpt:
        excerpt = compact

    if len(excerpt) > max_chars:
        return excerpt[:max_chars].rstrip() + "..."

    return excerpt


def build_contextual_image_summary(context_text: str) -> str:
    """Build a chunk-grounded fallback summary for an extracted image."""

    context_excerpt = normalize_image_context(context_text)
    if not context_excerpt:
        return "Document image extracted from the uploaded file."

    return f"Document image related to: {context_excerpt}"


def is_generic_image_summary(summary: str) -> bool:
    """Detect low-information image summaries."""

    normalized = summary.strip().lower()
    return normalized in {
        "",
        "document image",
        "retrieved image",
        "document image extracted from the uploaded file.",
    }


def summarize_image_for_retrieval(
    image_payload: bytes,
    *,
    mime_type: str,
    context_text: str,
) -> str:
    """Create a retrieval-oriented summary for one extracted image."""

    data_url = build_data_url(mime_type, image_payload)

    llm = ChatOpenAI()
    context_excerpt = normalize_image_context(context_text, max_chars=1200)
    if not context_excerpt:
        context_excerpt = "No nearby text available."
    prompt = [
        {
            "type": "text",
            "text": (
                "Summarize this document image for retrieval.\n"
                "Focus on what kind of image it is and how it relates to the nearby document text.\n"
                "Call out if it is mostly a logo, header mark, signature, stamp, photo, chart, table, or form section.\n"
                "Mention any clearly visible labels or fields when obvious.\n"
                "If it appears decorative or low-value, say that clearly and tie it to the nearby section.\n"
                f"Nearby document text:\n{context_excerpt}\n\n"
                "Return one concise sentence."
            ),
        },
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    response = llm.invoke([{"role": "user", "content": prompt}])
    content = response.content
    if isinstance(content, str):
        summary = content.strip()
        if not is_generic_image_summary(summary):
            return summary
        return build_contextual_image_summary(context_text)
    if isinstance(content, list):
        text_parts = [
            part.get("text", "").strip()
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        if text_parts:
            summary = " ".join(part for part in text_parts if part).strip()
            if not is_generic_image_summary(summary):
                return summary
            return build_contextual_image_summary(context_text)
    return build_contextual_image_summary(context_text)


def build_image_entry(
    asset: StoredImageAsset,
    image_payload: bytes,
    *,
    context_text: str,
) -> ImageEntry:
    """Build stored metadata for one extracted image."""

    try:
        summary = summarize_image_for_retrieval(
            image_payload,
            mime_type=asset.mime_type,
            context_text=context_text,
        )
    except Exception:
        summary = build_contextual_image_summary(context_text)

    if is_generic_image_summary(summary):
        summary = build_contextual_image_summary(context_text)

    return {
        "asset_id": asset.asset_id,
        "mime_type": asset.mime_type,
        "checksum": asset.checksum,
        "context_text": context_text.strip()[:800],
        "summary": summary,
    }


def separate_content_types(
    chunk: Element,
    *,
    user_id: str | None = None,
    document_id: str | None = None,
    asset_repository: ImageAssetRepository | None = None,
) -> ContentData:
    """Analyze what kind of content is in a chunk"""
    chunk_text = extract_chunk_text(chunk)
    content_data: ContentData = {
        "text": chunk_text,
        "tables": [],
        "images": [],
        "types": ["text"],
    }

    if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            if element_type == "Table":
                content_data["types"].append("table")
                table_html = getattr(element.metadata, "text_as_html", element.text)
                content_data["tables"].append(table_html)

            if element_type == "Image":
                content_data["types"].append("image")
                if asset_repository is None or user_id is None or document_id is None:
                    continue
                image_payload = decode_base64_image(
                    getattr(element.metadata, "image_base64", None)
                )
                if image_payload is None:
                    continue
                mime_type = getattr(element.metadata, "image_mime_type", None)
                if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
                    mime_type = "image/png"
                checksum = hashlib.sha256(image_payload).hexdigest()
                asset = asset_repository.store_asset(
                    user_id=user_id,
                    document_id=document_id,
                    mime_type=mime_type,
                    payload=image_payload,
                    checksum=checksum,
                )
                content_data["images"].append(
                    build_image_entry(
                        asset,
                        image_payload,
                        context_text=chunk_text,
                    )
                )

    content_data["types"] = list(set(content_data["types"]))
    return content_data


def create_ai_enhanced_content(content_data: ContentData):
    "Create AI enahnced Summary for mixed content"

    llm = ChatOpenAI()
    texts = content_data.get("text")
    tables = content_data.get("tables")
    images = content_data.get("images")

    prompt_temp = PromptTemplate.from_template("""
    You are creating a searchable description for mixed content, for the purpose of document retrieval
    CONTENT TO ANALYZE:
    {text_prompt}

    {table_prompt}
    """)

    text_prompt = f"""
    TEXT: \n {texts} \n
    """

    table_prompt = ""
    table_prompt += "\n TABLES: \n"
    for i, table in enumerate(tables):
        table_prompt += f"Table {i + 1}: {table} \n"

    image_prompt = "\n IMAGES: \n"
    for image_entry in images:
        if not isinstance(image_entry, dict):
            continue
        image_prompt += (
            f"{image_entry.get('asset_id', 'image')}: "
            f"{image_entry.get('summary', 'Document image')} "
            f"(mime={image_entry.get('mime_type')})\n"
        )

    table_prompt += """
            YOUR TASK:
            Generate a comprehensive, searchable description that covers:

            1. Key facts, numbers, and data points from text and tables
            2. Main topics and concepts discussed  
            3. Questions this content could answer
            4. Visual content analysis (charts, diagrams, patterns in images)
            5. Alternative search terms users might use

            Make it detailed and searchable - prioritize findability over brevity.

            SEARCHABLE DESCRIPTION:"""

    summarizer = prompt_temp | llm

    res = summarizer.invoke(
        {
            "text_prompt": text_prompt,
            "table_prompt": table_prompt + image_prompt,
        }
    )

    content = res.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            part.get("text", "").strip()
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        return " ".join(part for part in text_parts if part).strip()
    return str(content)


def summarize_chunk(
    chunk: Element,
    *,
    current_chunk: int,
    total_chunks: int,
    user_id: str | None = None,
    document_id: str | None = None,
    asset_repository: ImageAssetRepository | None = None,
    logger: Callable[[IngestionEvent], None] | None = None,
) -> Document:
    emit_log(logger, f"Summarizing chunk {current_chunk}/{total_chunks}")

    content_data = separate_content_types(
        chunk,
        user_id=user_id,
        document_id=document_id,
        asset_repository=asset_repository,
    )

    emit_log(logger, f"Types found: {content_data['types']}")
    emit_log(
        logger,
        f"Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}",
    )
    if content_data["images"]:
        image_debug = [
            {
                "asset_id": image.get("asset_id"),
                "summary": image.get("summary"),
            }
            for image in content_data["images"]
            if isinstance(image, dict)
        ]
        print(
            f"[RAG Images] chunk {current_chunk}/{total_chunks} extracted: "
            f"{json.dumps(image_debug, ensure_ascii=True)}",
            flush=True,
        )

    emit_log(logger, "Creating AI enhance summary")
    tables = content_data.get("tables")
    images = content_data.get("images")
    texts = content_data.get("text") or ""
    if tables or images:
        try:
            enhanced_content = create_ai_enhanced_content(content_data)
        except Exception as e:
            emit_log(logger, f"AI summary failed: {e}")
            enhanced_content = content_data.get("text")
    else:
        emit_log(logger, "Using raw text, no images and tables found")
        enhanced_content = content_data["text"] or ""

    metadata = {
        "original_content": json.dumps(
            {
                "raw_text": texts or "",
                "tables_html": tables or [],
                "image_entries": images or [],
            }
        )
    }
    if user_id is not None:
        metadata["user_id"] = user_id
    if document_id is not None:
        metadata["document_id"] = document_id

    page_content = enhanced_content if isinstance(enhanced_content, str) else texts
    return Document(page_content=page_content or texts, metadata=metadata)


def summarize_chunks(
    chunks: list[Element],
    logger: Callable[[IngestionEvent], None] | None = None,
) -> list[Document]:
    """Process all chunks"""
    emit_log(logger, "Processing chunks")

    langchain_docs = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        langchain_docs.append(
            summarize_chunk(
                chunk,
                current_chunk=current_chunk,
                total_chunks=total_chunks,
                logger=logger,
            )
        )

    emit_log(logger, f"Processed {len(langchain_docs)} chunks")
    return langchain_docs


async def stream_ingest_pdf(
    path: str | Path,
    *,
    file_name: str | None = None,
    user_id: str | None = None,
    document_id: str | None = None,
    asset_repository: ImageAssetRepository | None = None,
    storage: PGVectorStore | None = None,
    vector_service: PostgresVectorService | None = None,
    store_name: str = "raggidy_docs",
) -> AsyncIterator[IngestionEvent]:
    resolved_path = Path(path)
    resolved_name = file_name or resolved_path.name
    event_queue: list[IngestionEvent] = []

    def logger(event: IngestionEvent) -> None:
        event_queue.append(event)

    yield build_ingestion_event(
        "file",
        file_name=resolved_name,
        elements=0,
        chunks=0,
        docs=0,
    )
    emit_log(logger, f"Preparing ingestion for {resolved_name}")
    if storage is None:
        if vector_service is None:
            raise RuntimeError("Vector service must be initialized before ingestion.")
        emit_log(logger, f"Using warmed store: {store_name}")
        storage = vector_service.get_store(store_name)
        emit_log(logger, "Store ready")

    while event_queue:
        yield event_queue.pop(0)

    elements = partition_pdf_doc(resolved_path, logger=logger)
    emit_stats(
        logger,
        file_name=resolved_name,
        elements=len(elements),
        chunks=0,
        docs=0,
    )
    while event_queue:
        yield event_queue.pop(0)

    await asyncio.sleep(0)

    chunks = create_chunks_by_title(elements, logger=logger)
    emit_stats(
        logger,
        file_name=resolved_name,
        elements=len(elements),
        chunks=len(chunks),
        docs=0,
    )
    while event_queue:
        yield event_queue.pop(0)

    await asyncio.sleep(0)

    total_chunks = len(chunks)
    docs: list[Document] = []
    for index, chunk in enumerate(chunks, start=1):
        emit_log(logger, f"Processing chunk {index} of {total_chunks}")
        docs.append(
            summarize_chunk(
                chunk,
                current_chunk=index,
                total_chunks=total_chunks,
                user_id=user_id,
                document_id=document_id,
                asset_repository=asset_repository,
                logger=logger,
            )
        )
        emit_stats(
            logger,
            file_name=resolved_name,
            elements=len(elements),
            chunks=len(chunks),
            docs=len(docs),
        )
        while event_queue:
            yield event_queue.pop(0)
        await asyncio.sleep(0)

    emit_log(logger, f"Storing {len(docs)} docs in {store_name}")
    await storage.aadd_documents(docs)
    emit_log(logger, f"Ingestion complete for {resolved_name}")
    emit_stats(
        logger,
        file_name=resolved_name,
        elements=len(elements),
        chunks=len(chunks),
        docs=len(docs),
    )
    while event_queue:
        yield event_queue.pop(0)

    yield build_ingestion_event(
        "done",
        text="Ingestion complete",
        file_name=resolved_name,
        elements=len(elements),
        chunks=len(chunks),
        docs=len(docs),
    )


async def main():
    docs = ["attention.pdf", "cv.pdf"]
    vector_service = PostgresVectorService(table_names=("raggidy_docs",))
    await vector_service.startup()

    try:
        for d in docs:
            async for _ in stream_ingest_pdf(
                d,
                file_name=d,
                vector_service=vector_service,
            ):
                pass
    finally:
        await vector_service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
