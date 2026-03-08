"""KG Embedder: Layer 1 application for generating vector embeddings for KG entities.

This module provides embedding infrastructure for converting entities and relations
into vector representations using LiteLLM's embedding API.

The architecture follows Mellea's Layer 1 pattern:
- Layer 1: KGEmbedder (this module) orchestrates embedding operations
- Layer 3: Can integrate with LLM session for embedding generation
- Layer 4: Uses GraphBackend for storing/retrieving entities

Example::

    import asyncio
    from mellea import start_session
    from mellea_contribs.kg import MockGraphBackend, Entity
    from mellea_contribs.kg.embedder import KGEmbedder

    async def main():
        session = start_session(backend_name="litellm", model_id="gpt-3.5-turbo")
        backend = MockGraphBackend()
        embedder = KGEmbedder(
            session=session,
            embedding_model="text-embedding-3-small",
            embedding_dimension=1536
        )

        # Create an entity
        entity = Entity(
            type="Movie",
            name="Avatar",
            description="A science fiction film directed by James Cameron",
            paragraph_start="Avatar is",
            paragraph_end="by Cameron."
        )

        # Generate embedding
        entity_with_embedding = await embedder.embed_entity(entity)
        assert entity_with_embedding.embedding is not None
        assert len(entity_with_embedding.embedding) == 1536

        # Find similar entities
        similar = await embedder.get_similar_entities(
            entity_with_embedding,
            [entity_with_embedding],  # Search against the same entity for demo
            top_k=1
        )
        assert len(similar) == 1

        await backend.close()

    asyncio.run(main())
"""

import logging
import math
from typing import Optional

from mellea import MelleaSession

from mellea_contribs.kg.graph_dbs.base import GraphBackend
from mellea_contribs.kg.models import Entity, Relation

logger = logging.getLogger(__name__)


class KGEmbedder:
    """Generates and manages vector embeddings for KG entities and relations.

    This is a Layer 1 application that orchestrates embedding operations.
    It uses LiteLLM's embedding API for generating vector representations.

    The class supports:
    - Embedding individual entities
    - Batch embedding of multiple entities
    - Finding similar entities by embedding distance (cosine similarity)
    - Persistence through GraphBackend
    """

    def __init__(
        self,
        session: MelleaSession,
        embedding_model: str = "text-embedding-3-small",
        embedding_dimension: int = 1536,
        backend: Optional[GraphBackend] = None,
    ):
        """Initialize the KG embedder.

        Args:
            session: MelleaSession for LLM operations
            embedding_model: Name of embedding model (LiteLLM compatible)
                Default: "text-embedding-3-small" (OpenAI model)
            embedding_dimension: Dimension of embedding vectors
                Default: 1536 (OpenAI's embedding size)
            backend: Optional GraphBackend for persisting embeddings
        """
        self.session = session
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.backend = backend

    async def embed_entity(
        self,
        entity: Entity,
        use_name: bool = True,
        use_description: bool = True,
    ) -> Entity:
        """Generate embedding for a single entity.

        Args:
            entity: The Entity to embed
            use_name: Include entity name in embedding text (default True)
            use_description: Include entity description in embedding text (default True)

        Returns:
            Entity with embedding field populated
        """
        # Build text to embed
        text_parts = []
        if use_name:
            text_parts.append(f"Name: {entity.name}")
        if use_description:
            text_parts.append(f"Description: {entity.description}")

        embed_text = " ".join(text_parts)
        logger.debug(f"Embedding entity: {entity.name}")

        # Generate embedding using LiteLLM
        try:
            embedding = await self._get_embedding(embed_text)
            entity.embedding = embedding
            logger.debug(f"Generated embedding for {entity.name} ({len(embedding)} dimensions)")
            return entity
        except Exception as e:
            logger.error(f"Failed to embed entity {entity.name}: {e}")
            raise

    async def embed_batch(
        self,
        entities: list[Entity],
        use_name: bool = True,
        use_description: bool = True,
        batch_size: int = 10,
    ) -> list[Entity]:
        """Generate embeddings for multiple entities in parallel batches.

        Args:
            entities: List of entities to embed
            use_name: Include entity name in embedding text
            use_description: Include entity description in embedding text
            batch_size: Number of entities to embed in parallel per batch

        Returns:
            List of entities with embeddings populated
        """
        logger.info(f"Embedding batch of {len(entities)} entities")
        embedded_entities = []

        # Process in batches
        for i in range(0, len(entities), batch_size):
            batch = entities[i : i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1} ({len(batch)} entities)")

            # Embed each entity in batch (could be parallelized further)
            for entity in batch:
                try:
                    embedded = await self.embed_entity(
                        entity,
                        use_name=use_name,
                        use_description=use_description,
                    )
                    embedded_entities.append(embedded)
                except Exception as e:
                    logger.warning(f"Skipping entity {entity.name} due to embedding error: {e}")
                    # Add entity without embedding
                    embedded_entities.append(entity)

        logger.info(f"Embedded {len(embedded_entities)} entities")
        return embedded_entities

    async def get_similar_entities(
        self,
        query_entity: Entity,
        candidate_entities: list[Entity],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> list[tuple[Entity, float]]:
        """Find similar entities by embedding distance.

        Uses cosine similarity to find entities most similar to the query entity.

        Args:
            query_entity: Entity to query (must have embedding)
            candidate_entities: List of entities to search (must all have embeddings)
            top_k: Number of top similar entities to return
            similarity_threshold: Minimum similarity score (0-1) to include results

        Returns:
            List of (Entity, similarity_score) tuples sorted by similarity (highest first)

        Raises:
            ValueError: If query_entity or candidates don't have embeddings
        """
        if query_entity.embedding is None:
            raise ValueError("Query entity must have embedding")

        if not candidate_entities:
            return []

        # Compute similarity scores
        similarities = []
        for candidate in candidate_entities:
            if candidate.embedding is None:
                logger.warning(f"Skipping candidate {candidate.name} (no embedding)")
                continue

            similarity = self._cosine_similarity(query_entity.embedding, candidate.embedding)

            if similarity >= similarity_threshold:
                similarities.append((candidate, similarity))

        # Sort by similarity (highest first) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        result = similarities[:top_k]

        logger.debug(
            f"Found {len(result)} similar entities (threshold={similarity_threshold}, top_k={top_k})"
        )
        return result

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score between -1 and 1 (typically 0 to 1 for embeddings)
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same dimension")

        if not vec1 or not vec2:
            return 0.0

        # Compute dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Compute magnitudes
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using LiteLLM API.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding

        Raises:
            Exception: If embedding API call fails
        """
        try:
            # Use LiteLLM's embedding API through litellm.embedding()
            import litellm

            response = await litellm.aembedding(
                model=self.embedding_model,
                input=text,
            )

            # Extract embedding from response
            if isinstance(response, dict) and "data" in response:
                embedding = response["data"][0]["embedding"]
            else:
                # Handle different response formats
                embedding = response[0]["embedding"] if isinstance(response, list) else response

            return embedding
        except Exception as e:
            logger.error(f"Embedding API error: {e}")
            raise

    async def close(self):
        """Close connections and cleanup resources."""
        if self.backend:
            await self.backend.close()


__all__ = ["KGEmbedder"]
