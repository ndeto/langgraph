from langchain_classic.tools import BaseTool
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

from atlasai.config.sys_config import get_env

DOC_DB_PATH = get_env("PGVECTOR_CONNECTION")
def tool_store(tools_list: list[BaseTool]) -> PGVector:
    descriptions = []
    metadata = []
    tool_docs = []

    for t in tools_list:
        descriptions.append(f"Tool Name: {t.name}. Description: {t.description}")
        metadata.append(
            {
                "name": t.name,
                "parameters": t.args,
            }
        )

        record = {"name": t.name, "description": t.description, "parameters": t.args}

        tool_docs.append(Document(page_content=t.description, metadata=record))

    tool_store = PGVector(
        embeddings=OpenAIEmbeddings(),
        connection=DOC_DB_PATH,
        collection_name="tools",
        collection_metadata={"hnsw:space": "cosine"},
        use_jsonb=True,
    )

    tool_store.add_documents(tool_docs)

    return tool_store
