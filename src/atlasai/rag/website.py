from pydoc import doc

from langchain_unstructured import UnstructuredLoader
from atlasai.store.rag_retriever import get_store


class WebsiteRAG:
    def __init__(self, websites: list[str]) -> None:
        self.websites = websites
        self.store = get_store("websites")
        print(f"\n Initilized with {websites}! \n")

    def rag_sites(self):
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

            self.store.add_documents(docs)

            print(f"Done for {l}\n")


def main():
    links = ["https://ndeto.eth.limo"]

    ragger = WebsiteRAG(websites=links)
    ragger.rag_sites()

# if __name__ == "__main__":
#     main()