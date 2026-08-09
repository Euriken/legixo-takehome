"""
LangGraph nodes for the ask flow.

State shape (TypedDict, see `GraphState` below):
    question         - the current question text used for retrieval (may be
                        rewritten mid-graph)
    original_question - the user's original question, kept for the final answer
    chunks           - list of retrieved chunk dicts (text + metadata)
    loop_count       - how many retrieval attempts have happened
    grading_sufficient - bool set by grade_documents, drives the branch
    answer           - final answer text
    citations        - list of {source_file, chunk_id} used in the answer
"""

from typing import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from app.config import settings
from app.pinecone_client import get_index


class Citation(TypedDict):
    source_file: str
    chunk_id: str


class GraphState(TypedDict):
    question: str
    original_question: str
    chunks: list[dict]
    loop_count: int
    grading_sufficient: bool
    answer: str
    citations: list[Citation]


# ---- structured-output schemas for the LLM calls ----


class GradeResult(BaseModel):
    sufficient: bool = Field(
        description="True if the retrieved chunks contain enough information "
        "to answer the question accurately. False if they're irrelevant, "
        "off-topic, or too thin."
    )
    reason: str = Field(description="One sentence explaining the judgment.")


class AnswerResult(BaseModel):
    answer: str = Field(
        description="The answer to the question, grounded ONLY in the "
        "provided chunks. If the chunks don't actually support an answer, "
        "say so instead of guessing."
    )
    used_chunk_ids: list[str] = Field(
        description="chunk_id values (from the provided chunks) that were "
        "actually used to construct the answer. Empty list if none were "
        "usable."
    )


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key, temperature=0)


def _embedder() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


# ---- nodes ----


def retrieve(state: GraphState) -> dict:
    """Embed the current question, query Pinecone top-k, attach chunks."""
    index = get_index()
    vector = _embedder().embed_query(state["question"])

    result = index.query(
        vector=vector,
        top_k=settings.retrieval_top_k,
        include_metadata=True,
    )

    chunks = [
        {
            "chunk_id": match["metadata"].get("chunk_id"),
            "source_file": match["metadata"].get("source_file"),
            "chunk_index": match["metadata"].get("chunk_index"),
            "text": match["metadata"].get("chunk_text"),
            "score": match.get("score"),
        }
        for match in result.get("matches", [])
    ]
    return {"chunks": chunks}


def grade_documents(state: GraphState) -> dict:
    """LLM judges whether the retrieved chunks are sufficient to answer."""
    if not state["chunks"]:
        return {"grading_sufficient": False}

    context = "\n\n".join(
        f"[{c['chunk_id']}] ({c['source_file']}): {c['text']}" for c in state["chunks"]
    )
    prompt = (
        f"Question: {state['original_question']}\n\n"
        f"Retrieved chunks:\n{context}\n\n"
        "Are these chunks sufficient to answer the question accurately? "
        "Judge strictly -- if they're off-topic or too thin, say False."
    )
    result: GradeResult = _llm().with_structured_output(GradeResult).invoke(prompt)
    return {"grading_sufficient": result.sufficient}


def rewrite_query(state: GraphState) -> dict:
    """LLM rewrites the question for a retry, and bumps the loop counter."""
    prompt = (
        f"Original question: {state['original_question']}\n"
        f"Previous search query: {state['question']}\n\n"
        "The previous search did not return good enough results. Rewrite "
        "the search query with different phrasing, synonyms, or angle to "
        "improve retrieval. Return ONLY the rewritten query text."
    )
    rewritten = _llm().invoke(prompt).content.strip()
    return {
        "question": rewritten,
        "loop_count": state["loop_count"] + 1,
    }


def generate_answer(state: GraphState) -> dict:
    """LLM answers strictly from retrieved chunks, with citations."""
    context = "\n\n".join(
        f"[{c['chunk_id']}] ({c['source_file']}): {c['text']}" for c in state["chunks"]
    )
    prompt = (
        f"Question: {state['original_question']}\n\n"
        f"Retrieved chunks:\n{context}\n\n"
        "Answer the question using ONLY information in these chunks. "
        "Do not use outside knowledge. If the chunks don't actually "
        "contain the answer, say you cannot find it in the documents. "
        "List which chunk_ids you actually used."
    )
    result: AnswerResult = _llm().with_structured_output(AnswerResult).invoke(prompt)

    used_ids = set(result.used_chunk_ids)
    citations = [
        {"source_file": c["source_file"], "chunk_id": c["chunk_id"]}
        for c in state["chunks"]
        if c["chunk_id"] in used_ids
    ]
    return {"answer": result.answer, "citations": citations}


def answer_not_found(state: GraphState) -> dict:
    """Deterministic terminal node -- no LLM call, no hallucination risk."""
    return {
        "answer": "I cannot find this in the provided documents.",
        "citations": [],
    }


# ---- branch condition (used by the graph) ----


def route_after_grading(state: GraphState) -> str:
    if state["grading_sufficient"]:
        return "generate_answer"
    if state["loop_count"] >= settings.max_retrieval_loops:
        return "answer_not_found"
    return "rewrite_query"
