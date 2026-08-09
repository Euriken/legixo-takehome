"""
FastAPI app exposing the Q&A API.

Endpoints:
    POST /ask     - {"question": "..."} -> {"answer", "citations", "trace"}
    POST /ingest  - triggers corpus ingestion into Pinecone (optional
                    alternative to running `python -m app.ingest` directly)
    GET  /health  - basic liveness check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.graph import run_ask
from app.ingest import run_ingest

app = FastAPI(title="Legixo Take-Home: Document Q&A API")


class AskRequest(BaseModel):
    question: str


class Citation(BaseModel):
    source_file: str
    chunk_id: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    trace: dict


class IngestRequest(BaseModel):
    clear: bool = False


class IngestResponse(BaseModel):
    status: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    try:
        final_state = run_ask(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ask failed: {e}")

    return AskResponse(
        answer=final_state["answer"],
        citations=final_state["citations"],
        trace={
            "loop_count": final_state["loop_count"],
            "grading_sufficient": final_state["grading_sufficient"],
            "chunks_considered": [
                {"chunk_id": c["chunk_id"], "source_file": c["source_file"], "score": c["score"]}
                for c in final_state["chunks"]
            ],
        },
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    try:
        run_ingest(clear=request.clear)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingest failed: {e}")
    return IngestResponse(status="ok")
