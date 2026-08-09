"""
Ingest pipeline: read files from CORPUS_DIR -> chunk -> embed -> upsert to
Pinecone with deterministic point IDs.

Deterministic IDs (sha256 of "relative_path::chunk_index", truncated) mean
re-running ingest on an unchanged corpus overwrites the same vectors instead
of duplicating them. If a file shrinks (fewer chunks than before), old
leftover chunks for that file could remain as orphans -- use `--clear` to
wipe the whole index first if you want a guaranteed-clean rebuild.

Run:
    python -m app.ingest            # incremental upsert
    python -m app.ingest --clear    # wipe index, then ingest fresh
"""

import argparse
import hashlib
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.pinecone_client import get_index

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def read_corpus_files(corpus_dir: str) -> list[tuple[str, str]]:
    """Returns list of (relative_path, file_text) for every supported file."""
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            rel_path = str(path.relative_to(root))
            text = path.read_text(encoding="utf-8", errors="ignore")
            files.append((rel_path, text))
    return files


def chunk_file(rel_path: str, text: str) -> list[dict]:
    """Splits one file's text into chunks with metadata attached."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    pieces = splitter.split_text(text)

    chunks = []
    for i, piece in enumerate(pieces):
        chunk_id = hashlib.sha256(f"{rel_path}::{i}".encode()).hexdigest()[:24]
        chunks.append(
            {
                "id": chunk_id,
                "text": piece,
                "source_file": rel_path,
                "chunk_index": i,
            }
        )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Adds an 'embedding' key to each chunk dict, batching the API calls."""
    embedder = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    texts = [c["text"] for c in chunks]

    batch_size = 100
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(embedder.embed_documents(batch))

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def upsert_chunks(chunks: list[dict]) -> int:
    """Upserts all chunks to Pinecone. Returns count upserted."""
    index = get_index()

    records = [
        {
            "id": c["id"],
            "values": c["embedding"],
            "metadata": {
                "chunk_id": c["id"],
                "source_file": c["source_file"],
                "chunk_index": c["chunk_index"],
                "chunk_text": c["text"],
            },
        }
        for c in chunks
    ]

    batch_size = 100
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        index.upsert(vectors=batch)

    return len(records)


def run_ingest(clear: bool = False) -> None:
    if clear:
        index = get_index()
        print(f"Clearing index '{settings.pinecone_index_name}' before ingest...")
        index.delete(delete_all=True)

    print(f"Reading corpus from '{settings.corpus_dir}'...")
    files = read_corpus_files(settings.corpus_dir)
    print(f"Found {len(files)} file(s).")

    all_chunks: list[dict] = []
    for rel_path, text in files:
        file_chunks = chunk_file(rel_path, text)
        all_chunks.extend(file_chunks)
        print(f"  {rel_path}: {len(file_chunks)} chunk(s)")

    if not all_chunks:
        print("No chunks to ingest. Check CORPUS_DIR and file extensions.")
        return

    print(f"Embedding {len(all_chunks)} chunk(s)...")
    all_chunks = embed_chunks(all_chunks)

    print("Upserting to Pinecone...")
    count = upsert_chunks(all_chunks)
    print(f"Done. Upserted {count} vector(s) into '{settings.pinecone_index_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest corpus into Pinecone.")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing vectors in the index before ingesting.",
    )
    args = parser.parse_args()
    run_ingest(clear=args.clear)
