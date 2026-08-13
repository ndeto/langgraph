# AtlasAI

Disclaimer: This agent's LangGraph, Python, FastAPI, retrieval, and memory backend was engineered manually. AI assistance was used only for the frontend UI scaffold.

Find a full article on this project on [medium](https://medium.com/@ndeto/building-ai-agents-graph-loops-memory-retrieval-rag-bf2490930834)

AtlasAI is a long running conversational AI agent built with FastAPI, LangGraph, LangChain, and Postgres. The project focuses on a practical agent architecture. It features:

- graph-based orchestration using langGraph
- real retrieval infrastructure with Postgres hybrid search
- memory management beyond short chat history
- ingestion pipelines for both documents and websites
- an HTTP service layer that can evolve into a deployable backend



## What It Does

- Runs a LangGraph-based conversational agent behind a FastAPI API and Vite FrontEnd.
- Uses hybrid search retrieval with `PGVectorStore` on Postgres and `pgvector`.
- Supports advanced document ingestion using [Unstructured](https://unstructured.io) libraries
- Ingests indexed website content into a searchable store for profile and biography lookups.
- Maintains long-term conversational memory with LangMem and a background memory update flow.
- Persists graph state with a Postgres checkpointer for thread-aware conversations.

## Architecture

AtlasAI is split into a few main layers:

```text
                    +--------------------------------+
                    | FastAPI HTTP boundary          |
                    | src/atlasai/web/main.py        |
                    +---------------+----------------+
                                    |
                                    v
                    +--------------------------------+
                    | Agent graph service            |
                    | src/atlasai/service/           |
                    | graph_service.py               |
                    +---------------+----------------+
                                    |
                  +-----------------+-----------------+
                  |                                   |
                  v                                   v
+--------------------------------+   +--------------------------------+
| Hybrid retrieval store         |   | Ingestion pipelines            |
| src/atlasai/store/             |   | src/atlasai/rag/               |
| hybrid_store.py                |   | rag_ingestion.py               |
|                                |   | website.py                     |
+---------------+----------------+   +---------------+----------------+
                |                                    |
                v                                    v
+--------------------------------+   +--------------------------------+
| Postgres + pgvector            |   | PDFs and website HTML          |
| PGVectorStore + hybrid search  |   | partition, chunk, summarize    |
+--------------------------------+   +--------------------------------+
```


## Retrieval and Memory

This project uses a hybrid search retrieval setup rather than vector-only search.

- Dense retrieval is handled by [`PGVectorStore`](https://docs.langchain.com/oss/python/integrations/vectorstores/pgvectorstore).
- Keyword retrieval is handled by Postgres full-text search through a `tsvector` column.
- Result fusion uses [reciprocal rank fusion](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking) (`RRF`).

The agent also maintains memory in two forms:

- LangGraph checkpointing for thread-level conversational state.
- Long-term memory tools provided by [LangMem](https://langchain-ai.github.io/langmem/)

## Document Ingestion

The document pipeline is designed for more than plain text extraction.

- PDFs are partitioned with Unstructured.
- Chunks are created with title-aware chunking.
- Tables are preserved as HTML.
- Image payloads can be materialized to disk and referenced during summarization.
- Chunks are converted into searchable documents before being embedded and stored.

This makes the RAG path closer to a real ingestion workflow than a simple text splitter demo.

## Stack

- Python 3.11+
- FastAPI
- LangGraph
- LangChain
- `langchain-postgres`
- PostgreSQL + `pgvector`
- Unstructured
- OpenAI-compatible models

## Local Setup

Install dependencies:

```bash
uv sync
```

Start Postgres with `pgvector`:

```bash
docker compose -f docker-compose.postgres.yml up -d
```

Create your environment file from `.env.example` and set at least:

```bash
MODEL_PROVIDER=
MODEL=
MODEL_PROVIDER_BASE_URL=
MODEL_API_KEY=
SOUL_PATH=
SYSTEM_PROMPT_PATH=
CG_API_KEY=
DB_CONN=postgresql://atlasai:atlasai@localhost:55433/atlasai
PGVECTOR_CONNECTION=postgresql+psycopg://atlasai:atlasai@localhost:55433/atlasai
```

## Running

Start the app:

```bash
./scripts/start_web.sh
```

The script runs migrations, builds the frontend into `src/atlasai/web/static`, starts FastAPI, and starts the ingestion worker.

Then open the UI in your browser:

```text
http://127.0.0.1:8000
```

## Ingestion Workflows

Load PDF documents into the hybrid store:

```bash
uv run python -m atlasai.rag.rag_ingestion
```

Load website content into the website store:

```bash
uv run python -m atlasai.rag.website
```
