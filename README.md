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

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

## Example request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What court heard the dispute between the parties?"}'
```

Response shape:

```json
{
  "answer": "...",
  "citations": [
    {"source_file": "some_case.md", "chunk_id": "a1b2c3..."}
  ],
  "trace": {
    "loop_count": 0,
    "grading_sufficient": true,
    "chunks_considered": [
      {"chunk_id": "a1b2c3...", "source_file": "some_case.md", "score": 0.87}
    ]
  }
}
```

You can also trigger ingestion via the API instead of the CLI:

```bash
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"clear": false}'
```

## Self-test / eval

`eval/test_cases.json` holds the question set (10-15 cases, including at
least one out-of-corpus check). **Currently placeholder content** — swap in
real questions once you've downloaded the sample corpus / gold set from the
assignment page and ingested it (see `eval/test_cases.json` for the schema).

With the API running (`uvicorn app.main:app --reload`):

```bash
python run_eval.py
```

This fires every question at `/ask`, checks whether the expected source
file shows up in the returned citations (or, for out-of-corpus cases,
that zero citations came back), prints pass/fail per case, and writes
full results to `eval/results.json`.

## What's skipped so far

- `eval/test_cases.json` still has placeholder questions — needs to be
  filled in with real questions against the actual sample corpus once
  it's downloaded and ingested. Everything else (ingest, LangGraph flow,
  FastAPI endpoints, eval harness itself) is built and working.
