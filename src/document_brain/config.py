"""Application configuration loaded from environment variables.

Settings are read once at startup and exposed via the `settings` singleton.
Values come from environment variables, optionally loaded from a `.env` file
in the project root. The `.env` file is git-ignored; `.env.example` documents
the required schema.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. All fields are required unless a default is set."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    qdrant_url: str = Field(description="Full URL of the Qdrant cluster, including port.")
    qdrant_api_key: str = Field(description="API key with read/write access to the collection.")
    qdrant_collection: str = Field(
        default="document_brain",
        description="Name of the Qdrant collection holding document chunks.",
    )

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model ID for the sentence-transformer embedder.",
    )
    embedding_dimensions: int = Field(
        default=384,
        description="Output dimensions of the embedding model. Must match collection config.",
    )
    anthropic_api_key: str = Field(description="API key for Claude.")
    llm_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Anthropic model ID for answer generation.",
    )
    max_tokens: int = Field(default=1024, description="Max tokens in LLM response.")
    similarity_threshold: float = Field(
        default=0.3, description="Minimum cosine similarity for a chunk to be returned."
    )
    default_top_k: int = Field(default=5, description="Default number of chunks to retrieve.")


settings = Settings()  # type: ignore[call-arg]
