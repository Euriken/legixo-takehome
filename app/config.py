"""
Central app configuration. Reads from .env via pydantic-settings.
Every other module should import `settings` from here instead of
calling os.environ directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str = "sk-dummy-replace-me"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Pinecone
    pinecone_api_key: str = "pcsk-dummy-replace-me"
    pinecone_index_name: str = "legixo-corpus"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # App behavior
    corpus_dir: str = "corpus"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 5
    max_retrieval_loops: int = 2


settings = Settings()
