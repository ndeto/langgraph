import json
from typing import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_unstructured.document_loaders import Element
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers.ensemble import EnsembleRetriever
load_dotenv()


def partition_document(path: str):
    """Partitions the document"""
    print("Partitioning Document")
    elements = partition_pdf(
        filename=path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
    )

    print(f"Found {len(elements)} elements")
    return elements


def create_chunks_by_title(elements: list[Element]):
    """Create Intelligent chunks"""
    print("Creating smart chunks")
    chunks = chunk_by_title(
        elements=elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500,
    )

    print(f"Created {len(chunks)} chunks!")
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
                content_data["images"].append(element.metadata.image_base64)

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
            "table_prompt": table_prompt,
        }
    )

    return res.content


def summarize_chunks(chunks: list[Element]) -> list[Document]:
    """Process all chunks"""
    print("Processing Chunks")

    langchain_docs = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"Summarizing chunk {current_chunk}/{total_chunks}")

        content_data = separate_content_types(chunk)

        print(f"Types found: {content_data['types']}")
        print(f"Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")

        # Create AI enhanced Summaries
        print("Creating AI enchance summary")
        tables = content_data.get("tables")
        images = content_data.get("images")
        texts = content_data.get("text")
        if tables or images:
            try:
                enhanced_content = create_ai_enhanced_content(content_data)
            except Exception as e:
                print(f"AI Summary failed: {e}")
                enhanced_content = content_data.get("text")
        else:
            print("Using raw text, no images and tables found")
            enhanced_content = content_data["text"] or ""

        

        doc = Document(
            page_content=enhanced_content or texts or "",
            metadata={
                "original_content": json.dumps(
                    {
                        "raw_text": texts or "",
                        "tables_html": tables or [],
                        "images": images or [],
                    }
                )
            },
        )

        langchain_docs.append(doc)
    print(f"Processed {len(langchain_docs)} chunks")
    return langchain_docs

def doc_store(persistent_db="db/chroma.db"):

    vector_store = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persistent_db,
        collection_metadata={"hnsw:space": "cosine"}
    )

    return vector_store.as_retriever(kwargs={"k": 3})


def main():
    docs = ["attention.pdf", "cv.pdf"]
    doc_retriever = doc_store()

    for d in docs:
        elements = partition_document(d)
        chunks = create_chunks_by_title(elements)
        docs = summarize_chunks(chunks)

        doc_retriever.add_documents(docs)

    res = doc_retriever.invoke("what is the transformer model architecture")
    return res

if __name__ == "__main__":
    main()
