from collections.abc import Iterable, Sequence
from logging import Logger

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.v2.hybrid_search_config import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from atlasai.config.sys_config import get_env

DEFAULT_VECTOR_TABLES = ("raggidy_docs", "websites")


class PostgresVectorService:
    """Owns warmed Postgres vector store handles."""

    def __init__(self, table_names: Iterable[str] | None = None) -> None:
        self.conn = get_env("PGVECTOR_CONNECTION")
        self.logger = Logger(name="postgres-vector-service")
        self.table_names = tuple(dict.fromkeys(table_names or DEFAULT_VECTOR_TABLES))
        self.hybrid_search_config = HybridSearchConfig(
            tsv_column="hybrid_text",
            fusion_function=reciprocal_rank_fusion,
            fusion_function_parameters={"rrf_k": 60, "fetch_top_k": 10},
        )
        self.engine: AsyncEngine | None = None
        self.pg_engine: PGEngine | None = None
        self._stores: dict[str, PGVectorStore] = {}

    async def startup(self) -> None:
        """Open the engine and warm configured vector stores."""

        if self.pg_engine is not None:
            return

        self.engine = create_async_engine(self.conn)
        self.pg_engine = PGEngine.from_engine(self.engine)

        for table_name in self.table_names:
            await self._init_store(table_name)

    async def shutdown(self) -> None:
        """Dispose the async engine."""

        self._stores.clear()

        if self.engine is not None:
            await self.engine.dispose()

        self.engine = None
        self.pg_engine = None

    def get_store(self, table_name: str) -> PGVectorStore:
        """Return a warmed vector store handle."""

        try:
            return self._stores[table_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Vector store '{table_name}' is not initialized."
            ) from exc

    async def aadd_documents(
        self,
        *,
        table_name: str,
        documents: Sequence[Document],
    ) -> list[str]:
        """Add documents to the configured table."""

        return await self.get_store(table_name).aadd_documents(list(documents))

    async def asimilarity_search(
        self,
        *,
        table_name: str,
        query: str,
        **kwargs,
    ) -> list[Document]:
        """Run similarity search against a warmed table."""

        return await self.get_store(table_name).asimilarity_search(query, **kwargs)

    async def adelete_by_user(self, *, table_name: str, user_id: str) -> None:
        """Delete all rows owned by the given user from the configured table."""

        if self.engine is None:
            raise RuntimeError("Vector service startup must run before use.")

        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    f"DELETE FROM {table_name} "
                    "WHERE langchain_metadata->>'user_id' = :user_id"
                ),
                {"user_id": user_id},
            )

    async def adelete_by_document(
        self,
        *,
        table_name: str,
        user_id: str,
        document_id: str,
    ) -> None:
        """Delete rows for one owned document before an idempotent retry."""

        if self.engine is None:
            raise RuntimeError("Vector service startup must run before use.")

        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    f"DELETE FROM {table_name} "
                    "WHERE langchain_metadata->>'user_id' = :user_id "
                    "AND langchain_metadata->>'document_id' = :document_id"
                ),
                {"user_id": user_id, "document_id": document_id},
            )

    async def _init_store(self, table_name: str) -> None:
        if self.pg_engine is None or self.engine is None:
            raise RuntimeError("Vector service startup must run before use.")

        async with self.engine.connect() as conn:
            existing_table = await conn.scalar(
                text("SELECT to_regclass(:table_name)"),
                {"table_name": table_name},
            )

        if existing_table is None:
            await self.pg_engine.ainit_vectorstore_table(
                table_name=table_name,
                vector_size=1536,
                hybrid_search_config=self.hybrid_search_config,
                store_metadata=True,
            )

        self._stores[table_name] = await PGVectorStore.create(
            engine=self.pg_engine,
            embedding_service=OpenAIEmbeddings(),
            table_name=table_name,
            hybrid_search_config=self.hybrid_search_config,
        )
