from langchain_classic.tools import BaseTool
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

from atlasai.config.sys_config import get_env

DOC_DB_PATH = get_env("PGVECTOR_CONNECTION")


def tool_store(tools_list: list[BaseTool]) -> PGVector:
    tool_docs = []
    unique_tools = {}

    for t in tools_list:
        unique_tools.setdefault(t.name, t)

    for t in unique_tools.values():
        record = {"name": t.name, "description": t.description, "parameters": t.args}

        tool_docs.append(
            Document(
                page_content=f"Tool name: {t.name}. Description: {t.description}",
                metadata=record,
            )
        )

    tool_store = PGVector(
        embeddings=OpenAIEmbeddings(),
        connection=DOC_DB_PATH,
        collection_name="atlas_tools",
        collection_metadata={"hnsw:space": "cosine"},
        use_jsonb=True,
    )

    return tool_store
