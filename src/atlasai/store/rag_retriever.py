from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

from atlasai.config.sys_config import get_env

DOC_DB_PATH = get_env("PGVECTOR_CONNECTION")

def get_store(collection: str):
     return PGVector(
            embeddings=OpenAIEmbeddings(),
            connection=DOC_DB_PATH,
            collection_name=collection,
            collection_metadata={"hnsw:space": "cosine"},
            use_jsonb=True,
        )

def doc_store():
    return get_store("my docs")


def doc_retriever():
    store = doc_store()
    return store.as_retriever()
