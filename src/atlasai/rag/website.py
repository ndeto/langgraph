import asyncio

from langchain_postgres import PGVectorStore
from langchain_unstructured import UnstructuredLoader

from atlasai.store.hybrid_store import get_or_create_store


class WebsiteRAG:
    def __init__(self, websites: list[str], store: PGVectorStore):
        self.websites = websites
        self.store: PGVectorStore = store
        print(f"\n Initilized with {websites}! \n")

    async def rag_sites(self):
        for l in self.websites:
            print(f"Ragging {l} \n")
            docs = UnstructuredLoader(
                web_url=l,
                chunking_strategy="by_title",
                max_characters=3000,
                new_after_n_chars=2400,
                combine_text_under_n_chars=500,
                include_orig_elements=True,
            ).load()

            print(f"{len(docs)} docs found for {l} \n")

            await self.store.aadd_documents(docs)

            print(f"Done for {l}\n")


async def main():
    links = ["https://ndeto.eth.limo"]

    store = await get_or_create_store("websites")

    ragger = WebsiteRAG(websites=links, store=store)
    
    await ragger.rag_sites()


if __name__ == "__main__":
    asyncio.run(main())
