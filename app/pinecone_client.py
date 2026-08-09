"""
Pinecone client + index management.

- get_index(): returns a ready-to-use Pinecone Index handle, creating the
  serverless index first if it doesn't exist yet.
- Index dimension/metric are fixed to match the embedding model in config.
"""

import time

from pinecone import Pinecone, ServerlessSpec

from app.config import settings

_pc: Pinecone | None = None


def get_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    return _pc


def ensure_index_exists() -> None:
    """Create the index if it doesn't exist. Safe to call repeatedly."""
    pc = get_client()
    existing = {idx["name"] for idx in pc.list_indexes()}
    if settings.pinecone_index_name in existing:
        return

    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=settings.embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        ),
    )

    # Wait for the index to be ready before returning.
    while True:
        desc = pc.describe_index(settings.pinecone_index_name)
        if desc.status.get("ready"):
            break
        time.sleep(1)


def get_index():
    """Returns a handle to the (already existing) index."""
    ensure_index_exists()
    pc = get_client()
    return pc.Index(settings.pinecone_index_name)
