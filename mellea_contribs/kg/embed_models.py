"""Pydantic models for KG embedding configuration.

This module provides configuration models for the embedding pipeline.
"""

from typing import Optional

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model and API.

    Specifies which embedding model to use and how to connect to it.
    """

    model: str = Field(
        default="text-embedding-3-small",
        description="Name of embedding model (LiteLLM compatible). "
        "Examples: 'text-embedding-3-small', 'text-embedding-3-large', "
        "'all-MiniLM-L6-v2' (HuggingFace), 'nomic-embed-text' (Ollama)",
    )

    api_base: Optional[str] = Field(
        default=None,
        description="API base URL for custom embedding service. "
        "If None, uses default LiteLLM routing",
    )

    api_key: Optional[str] = Field(
        default=None,
        description="API key for embedding service (if required). "
        "Can be set via environment variable",
    )

    dimension: int = Field(
        default=1536,
        description="Dimension of embedding vectors. "
        "Default 1536 for OpenAI text-embedding-3-small. "
        "384 for all-MiniLM-L6-v2, 768 for nomic-embed-text",
    )

    batch_size: int = Field(
        default=10,
        description="Number of entities to embed in parallel per batch",
    )


class EmbeddingResult(BaseModel):
    """Result of embedding a single entity or document.

    Contains the original text and its vector representation.
    """

    text: str = Field(description="Original text that was embedded")

    embedding: list[float] = Field(description="Vector embedding")

    model: str = Field(description="Model name used for embedding")

    dimension: int = Field(description="Dimension of the embedding vector")


class EmbeddingSimilarity(BaseModel):
    """Result of similarity search.

    Represents an entity matched by embedding similarity.
    """

    entity_id: str = Field(description="ID of the matched entity")

    entity_name: str = Field(description="Name of the matched entity")

    similarity_score: float = Field(
        description="Similarity score (0-1, where 1 is most similar)"
    )

    entity_type: Optional[str] = Field(
        default=None, description="Type/label of the entity"
    )


class EmbeddingStats(BaseModel):
    """Statistics about embedding operations.

    Tracks performance and results of batch embedding, including optional
    storage and index creation counts when used in a full pipeline.
    """

    total_entities: int = Field(description="Total entities processed")

    successful_embeddings: int = Field(description="Number of successfully embedded entities")

    failed_embeddings: int = Field(description="Number of entities that failed to embed")

    skipped_embeddings: int = Field(description="Number of entities skipped (e.g., already embedded)")

    average_embedding_time: float = Field(
        description="Average time (seconds) to embed one entity"
    )

    total_time: float = Field(description="Total time (seconds) for the batch")

    model_used: str = Field(description="Embedding model used")

    # Pipeline-level fields (populated by embed_and_store_all)
    dimension: int = Field(default=0, description="Embedding vector dimension")

    total_relations: int = Field(default=0, description="Total relations processed")

    successful_relation_embeddings: int = Field(
        default=0, description="Number of successfully embedded relations"
    )

    failed_relation_embeddings: int = Field(
        default=0, description="Number of relations that failed to embed"
    )

    entities_stored: int = Field(default=0, description="Embeddings stored back to the graph DB")

    relations_stored: int = Field(
        default=0, description="Relation embeddings stored back to the graph DB"
    )

    vector_indices_created: int = Field(
        default=0, description="Vector indices created for similarity search"
    )

    success: bool = Field(default=True, description="Whether the pipeline completed successfully")

    error_message: str = Field(default="", description="Error message if pipeline failed")


__all__ = [
    "EmbeddingConfig",
    "EmbeddingResult",
    "EmbeddingSimilarity",
    "EmbeddingStats",
]
