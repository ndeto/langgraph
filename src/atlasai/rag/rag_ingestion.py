import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVectorStore
from langchain_unstructured.document_loaders import Element
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from atlasai.config.sys_config import SysConfig, bootstrap_config
from atlasai.rag.image_payloads import persist_base64_image
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
    event: IngestionEvent = {"type": event_type, **extra}
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
    emit_log(logger, "Partitioning document")
    elements = partition_pdf(
        filename=str(path),
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )

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


def extract_chunk_text(chunk: Element) -> str:
    """Extract usable text from HTML or PDF chunks."""
    text = getattr(chunk, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    page_content = getattr(chunk, "page_content", None)
    if isinstance(page_content, str) and page_content.strip():
        return page_content.strip()

    return str(chunk).strip()


def separate_content_types(chunk: Element) -> ContentData:
    """Analyze what kind of content is in a chunk"""
    content_data: ContentData = {
        "text": extract_chunk_text(chunk),
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
                image_path = persist_base64_image(element.metadata.image_base64)
                if image_path:
                    content_data["images"].append(image_path)

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
    for i, image_path in enumerate(images):
        image_prompt += f"Image {i + 1}: {image_path} \n"

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

    return res.content


def summarize_chunk(
    chunk: Element,
    *,
    current_chunk: int,
    total_chunks: int,
    user_id: str | None = None,
    logger: Callable[[IngestionEvent], None] | None = None,
) -> Document:
    emit_log(logger, f"Summarizing chunk {current_chunk}/{total_chunks}")

    content_data = separate_content_types(chunk)

    emit_log(logger, f"Types found: {content_data['types']}")
    emit_log(
        logger,
        f"Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}",
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
                "image_paths": images or [],
            }
        )
    }
    if user_id is not None:
        metadata["user_id"] = user_id

    return Document(
        page_content=enhanced_content or texts,
        metadata=metadata,
    )


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
