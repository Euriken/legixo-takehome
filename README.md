# Legixo Thinklabs — Gen AI Take-Home

A small Q&A HTTP API over a legal-notes document corpus. Retrieval via
Pinecone, orchestration via LangGraph, served with FastAPI.

> **Status: work in progress.** This README is being filled in piece by
> piece as the project is built. See `docs/langgraph.md` for the current
> graph design.

## Stack

- Python 3.10+
- FastAPI (HTTP API)
- LangGraph (`StateGraph`) for the ask flow
- Pinecone (serverless) for vector search
- OpenAI (`gpt-4o-mini` for generation, `text-embedding-3-small` for embeddings)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# then fill in OPENAI_API_KEY and PINECONE_API_KEY in .env
```

## Ingest

Drop `.txt`/`.md` files into `corpus/` (the sample corpus works as-is), then:

```bash
python -m app.ingest
```

This chunks each file (`RecursiveCharacterTextSplitter`, size/overlap set via
`CHUNK_SIZE`/`CHUNK_OVERLAP` in `.env`), embeds chunks with
`text-embedding-3-small`, and upserts them into Pinecone with metadata
(`chunk_id`, `source_file`, `chunk_index`, `chunk_text`).

The index is created automatically on first run if it doesn't exist
(serverless, cosine metric, dimension matching the embedding model).

**Re-running ingest:** point IDs are deterministic
(`sha256("relative_path::chunk_index")`), so re-ingesting an unchanged
corpus just overwrites the same vectors — no duplicates. If a file shrinks
(fewer chunks than a previous version), old leftover chunks for that file
could remain as orphans; use `--clear` to wipe the whole index first for a
guaranteed-clean rebuild:

```bash
python -m app.ingest --clear
```

## Run the API

*(not yet built — instructions will go here)*

## Example request

*(not yet built — curl example will go here)*

## What's skipped so far

- Graph nodes, API routes, and eval harness are next. Ingest pipeline is
  done (see above).
