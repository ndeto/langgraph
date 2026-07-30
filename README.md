# AtlasAI

Disclaimer: AtlasAI's LangGraph, Python, FastAPI, retrieval, and memory backend was engineered manually. AI assistance was used only for the frontend UI scaffold.

AtlasAI is a production-leaning AI agent built with FastAPI, LangGraph, LangChain, and Postgres. It is meant to show applied agent engineering work:

- graph-based orchestration
- real retrieval infrastructure with Postgres hybrid search
- memory management beyond short chat history
- ingestion pipelines for both documents and websites
- an HTTP service layer that can evolve into a deployable backend


The project focuses on a practical agent architecture. It combines tool use, hybrid retrieval, long-term memory, and advanced document ingestion into one system that can be invoked over HTTP.

## What It Does

- Runs a LangGraph-based conversational agent behind a FastAPI API.
- Uses hybrid retrieval with `PGVectorStore` on Postgres and `pgvector`.
- Supports advanced document RAG with PDF partitioning, chunking, summarization, and metadata preservation.
- Ingests indexed website content into a searchable store for profile and biography lookups.
- Maintains long-term conversational memory with LangMem and a background memory update flow.
- Persists graph state with a Postgres checkpointer for thread-aware conversations.

## Architecture

AtlasAI is split into a few main layers:

- `src/atlasai/web/main.py`
  FastAPI entrypoint with the `/invoke` endpoint.
- `src/atlasai/service/graph_service.py`
  Main agent orchestration, tool routing, memory extraction, and graph execution.
- `src/atlasai/store/hybrid_store.py`
  Postgres-backed `PGVectorStore` setup with built-in hybrid search configuration.
- `src/atlasai/rag/rag_ingestion.py`
  PDF ingestion pipeline for partitioning, chunking, summarizing, and storing documents.
- `src/atlasai/rag/website.py`
  Website ingestion flow for loading and storing HTML content.

## Retrieval and Memory

This project uses a hybrid retrieval setup rather than vector-only search.

- Dense retrieval is handled by `PGVectorStore`.
- Keyword retrieval is handled by Postgres full-text search through a `tsvector` column.
- Result fusion uses reciprocal rank fusion (`RRF`).

The agent also maintains memory in two forms:

- LangGraph checkpointing for thread-level conversational state.
- Long-term memory tools and a background memory graph for durable user facts and preferences.

## Document RAG

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
uv run fastapi dev src/atlasai/web/main.py
```

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
