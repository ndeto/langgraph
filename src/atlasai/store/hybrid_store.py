from logging import Logger

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.v2.hybrid_search_config import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
)
from sqlalchemy.ext.asyncio import create_async_engine

from atlasai.config.sys_config import get_env


class HybridStore:
    def __init__(self) -> None:
        self.conn = get_env("PGVECTOR_CONNECTION")
        self.logger = Logger(name="logger")
        engine = create_async_engine(self.conn)
        self.pg_engine = PGEngine.from_engine(engine)
        self.hybrid_search_config = HybridSearchConfig(
            tsv_column="hybrid_text",
            fusion_function=reciprocal_rank_fusion,
            fusion_function_parameters={"rrf_k": 60, "fetch_top_k": 10},
        )

    async def _get_or_create_store(self, table_name: str) -> PGVectorStore:
        """Create a vector store with the provided name configured with hybrid search and returns the store"""
        # Try to create the vector store
        try:
            await self.pg_engine.ainit_vectorstore_table(
                table_name=table_name,
                vector_size=1536,
                hybrid_search_config=self.hybrid_search_config,
                store_metadata=True,
            )
        except Exception as e:
            self.logger.warning(f"Did not init vectore, probably exists. Error: {e}")

        # Return the store

        store = await PGVectorStore.create(
            engine=self.pg_engine,
            embedding_service=OpenAIEmbeddings(),
            table_name=table_name,
            hybrid_search_config=self.hybrid_search_config,
        )

        return store

async def get_or_create_store(table_name: str) -> PGVectorStore:
    hs = HybridStore()
    return await hs._get_or_create_store(table_name=table_name)
