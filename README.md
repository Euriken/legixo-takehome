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

*(not yet built — instructions will go here)*

## Run the API

*(not yet built — instructions will go here)*

## Example request

*(not yet built — curl example will go here)*

## What's skipped so far

- Everything except project scaffold and config. Ingest, graph nodes, API
  routes, and eval harness are next.
