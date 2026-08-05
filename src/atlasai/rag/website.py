import asyncio

from langchain_postgres import PGVectorStore
from langchain_unstructured import UnstructuredLoader

from atlasai.store.hybrid_store import PostgresVectorService


class WebsiteRAG:
    def __init__(self, websites: list[str], store: PGVectorStore):
        self.websites = websites
        self.store: PGVectorStore = store
        print(f"\n Initilized with {websites}! \n")

    async def rag_sites(self):
        for website in self.websites:
            print(f"Ragging {website} \n")
            docs = UnstructuredLoader(
                web_url=website,
                chunking_strategy="by_title",
                max_characters=3000,
                new_after_n_chars=2400,
                combine_text_under_n_chars=500,
                include_orig_elements=True,
            ).load()

            print(f"{len(docs)} docs found for {website} \n")

            await self.store.aadd_documents(docs)

            print(f"Done for {website}\n")


async def main():
    links = ["https://ndeto.eth.limo"]
    vector_service = PostgresVectorService(table_names=("websites",))
    await vector_service.startup()

    try:
        store = vector_service.get_store("websites")
        ragger = WebsiteRAG(websites=links, store=store)
        await ragger.rag_sites()
    finally:
        await vector_service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
