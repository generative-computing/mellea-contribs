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
import time
from datetime import datetime
from typing import Optional

from mellea import MelleaSession

from mellea_contribs.kg.embed_models import EmbeddingStats
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
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 10,
        backend: Optional[GraphBackend] = None,
    ):
        """Initialize the KG embedder using individual parameters (Mellea Layer 1 pattern).

        Matches the pattern used by KGRag and KGPreprocessor with individual
        configuration parameters rather than a config object.

        Args:
            session: MelleaSession for LLM operations (required)
            model: Name of embedding model (LiteLLM compatible).
                Default: "text-embedding-3-small" (OpenAI model)
            dimension: Dimension of embedding vectors.
                Default: 1536 (OpenAI's embedding size)
            api_base: Optional API base URL for custom embedding service.
                If None, uses default LiteLLM routing
            api_key: Optional API key for embedding service
            batch_size: Number of entities to embed in parallel per batch.
                Default: 10
            backend: Optional GraphBackend for persisting embeddings
        """

        self.session = session
        self.embedding_model = model
        self.embedding_dimension = dimension
        self.api_base = api_base
        self.api_key = api_key
        self.batch_size_value = batch_size
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

    # ------------------------------------------------------------------
    # Neo4j pipeline: fetch / store / index
    # ------------------------------------------------------------------

    async def fetch_entities_from_neo4j(self) -> list[Entity]:
        """Fetch all entities from Neo4j as Entity objects.

        Returns:
            List of Entity objects from the graph, or empty list for non-Neo4j backends.
        """
        if not self.backend or getattr(self.backend, "backend_id", None) != "neo4j":
            return []

        cypher = """
        MATCH (n)
        RETURN n.name AS name, labels(n)[0] AS type
        LIMIT 100000
        """
        try:
            driver = getattr(self.backend, "_async_driver", None)
            if driver is None:
                return []
            async with driver.session() as session:
                result = await session.run(cypher)
                records = [r async for r in result]
            return [
                Entity(
                    type=r.get("type", "Unknown"),
                    name=r.get("name", ""),
                    description=f"Node of type {r.get('type')}",
                )
                for r in records
            ]
        except Exception as exc:
            logger.warning(f"Failed to fetch entities from Neo4j: {exc}")
            return []

    async def fetch_relations_from_neo4j(self) -> list[Relation]:
        """Fetch all relations from Neo4j as Relation objects.

        Returns:
            List of Relation objects from the graph, or empty list for non-Neo4j backends.
        """
        if not self.backend or getattr(self.backend, "backend_id", None) != "neo4j":
            return []

        cypher = """
        MATCH ()-[r]->()
        RETURN type(r) AS relation_type, id(r) AS rel_id
        LIMIT 100000
        """
        try:
            driver = getattr(self.backend, "_async_driver", None)
            if driver is None:
                return []
            async with driver.session() as session:
                result = await session.run(cypher)
                records = [r async for r in result]
            return [
                Relation(
                    relation_type=r.get("relation_type", "UNKNOWN"),
                    source_entity_type="Node",
                    source_entity_name=f"Relation_{r.get('rel_id')}",
                    target_entity_type="Node",
                    target_entity_name="Target",
                )
                for r in records
            ]
        except Exception as exc:
            logger.warning(f"Failed to fetch relations from Neo4j: {exc}")
            return []

    async def store_entity_embeddings(self, entities: list[Entity]) -> int:
        """Store entity embeddings back to Neo4j.

        Args:
            entities: Entities with populated ``embedding`` fields.

        Returns:
            Number of embeddings stored, or 0 for non-Neo4j backends.
        """
        if not self.backend or getattr(self.backend, "backend_id", None) != "neo4j":
            return 0

        cypher = """
        UNWIND $batch AS item
        MATCH (n {name: item.name})
        SET n.embedding = item.embedding
        RETURN count(n) AS updated
        """
        batch = [
            {"name": e.name, "embedding": getattr(e, "embedding", []) or []}
            for e in entities
        ]
        try:
            driver = getattr(self.backend, "_async_driver", None)
            if driver is None:
                return 0
            async with driver.session() as session:
                result = await session.run(cypher, batch=batch)
                record = await result.single()
                return record.get("updated", 0) if record else 0
        except Exception as exc:
            logger.warning(f"Failed to store entity embeddings: {exc}")
            return 0

    async def store_relation_embeddings(self, relations: list[Relation]) -> int:
        """Store relation embeddings back to Neo4j.

        Args:
            relations: Relations with populated ``embedding`` fields.

        Returns:
            Number of embeddings stored, or 0 for non-Neo4j backends.
        """
        if not self.backend or getattr(self.backend, "backend_id", None) != "neo4j":
            return 0

        cypher = """
        UNWIND $batch AS item
        MATCH ()-[r {type: item.relation_type}]->()
        SET r.embedding = item.embedding
        RETURN count(r) AS updated
        """
        batch = [
            {
                "relation_type": r.relation_type,
                "embedding": getattr(r, "embedding", []) or [],
            }
            for r in relations
        ]
        try:
            driver = getattr(self.backend, "_async_driver", None)
            if driver is None:
                return 0
            async with driver.session() as session:
                result = await session.run(cypher, batch=batch)
                record = await result.single()
                return record.get("updated", 0) if record else 0
        except Exception as exc:
            logger.warning(f"Failed to store relation embeddings: {exc}")
            return 0

    async def create_vector_indices(self) -> int:
        """Create Neo4j vector indices for embedding similarity search.

        Creates one index for entity nodes and one for relationship embeddings
        using the configured ``embedding_dimension``.

        Returns:
            Number of indices created, or 0 for non-Neo4j backends.
        """
        if not self.backend or getattr(self.backend, "backend_id", None) != "neo4j":
            return 0

        driver = getattr(self.backend, "_async_driver", None)
        if driver is None:
            return 0

        indices_created = 0
        dim = self.embedding_dimension
        index_queries = [
            f"""
            CREATE VECTOR INDEX IF NOT EXISTS entity_embedding_index
            FOR (n) ON (n.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """,
            f"""
            CREATE VECTOR INDEX IF NOT EXISTS relation_embedding_index
            FOR (r: RELATIONSHIP) ON (r.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """,
        ]
        try:
            async with driver.session() as session:
                for query in index_queries:
                    try:
                        await session.run(query)
                        indices_created += 1
                    except Exception as exc:
                        logger.debug(f"Vector index creation note: {exc}")
        except Exception as exc:
            logger.warning(f"Failed to create vector indices: {exc}")

        return indices_created

    async def embed_and_store_all(self, batch_size: int = 100) -> EmbeddingStats:
        """Run the full embedding pipeline: fetch → embed → store → index.

        Fetches entities and relations from the graph backend, generates
        embeddings for each, stores them back, and creates vector indices
        for similarity search.  For non-Neo4j backends the fetch/store/index
        steps are no-ops so the method can still be used in tests with the
        mock backend.

        Args:
            batch_size: Number of items to log progress at each interval.

        Returns:
            :class:`~mellea_contribs.kg.embed_models.EmbeddingStats` populated
            with counts for entities, relations, storage, and index creation.
        """
        t_start = time.monotonic()

        entities_embedded = entities_failed = entities_stored = 0
        relations_embedded = relations_failed = relations_stored = 0
        vector_indices = 0

        try:
            # --- entities ---------------------------------------------------
            logger.info("Embedding pipeline: fetching entities…")
            entities = await self.fetch_entities_from_neo4j()
            logger.info(f"  Fetched {len(entities)} entities")

            for i, entity in enumerate(entities):
                try:
                    await self.embed_entity(entity)
                    entities_embedded += 1
                except Exception:
                    entities_failed += 1
                if (i + 1) % batch_size == 0:
                    logger.info(f"  Embedded {i + 1}/{len(entities)} entities…")

            if entities_embedded:
                entities_stored = await self.store_entity_embeddings(entities)
                logger.info(f"  Stored {entities_stored} entity embeddings")

            # --- relations --------------------------------------------------
            logger.info("Embedding pipeline: fetching relations…")
            relations = await self.fetch_relations_from_neo4j()
            logger.info(f"  Fetched {len(relations)} relations")

            for i, relation in enumerate(relations):
                try:
                    text = f"Relation: {relation.relation_type}"
                    relation.embedding = await self._get_embedding(text)  # type: ignore[attr-defined]
                    relations_embedded += 1
                except Exception:
                    relations_failed += 1
                if (i + 1) % batch_size == 0:
                    logger.info(f"  Embedded {i + 1}/{len(relations)} relations…")

            if relations_embedded:
                relations_stored = await self.store_relation_embeddings(relations)
                logger.info(f"  Stored {relations_stored} relation embeddings")

            # --- vector indices ---------------------------------------------
            logger.info("Embedding pipeline: creating vector indices…")
            vector_indices = await self.create_vector_indices()
            logger.info(f"  Created {vector_indices} vector indices")

            total_time = time.monotonic() - t_start
            n_total = max(len(entities), 1)
            return EmbeddingStats(
                total_entities=len(entities),
                successful_embeddings=entities_embedded,
                failed_embeddings=entities_failed,
                skipped_embeddings=0,
                average_embedding_time=total_time / n_total,
                total_time=total_time,
                model_used=self.embedding_model,
                dimension=self.embedding_dimension,
                total_relations=len(relations),
                successful_relation_embeddings=relations_embedded,
                failed_relation_embeddings=relations_failed,
                entities_stored=entities_stored,
                relations_stored=relations_stored,
                vector_indices_created=vector_indices,
                success=True,
            )

        except Exception as exc:
            total_time = time.monotonic() - t_start
            logger.error(f"Embedding pipeline failed: {exc}")
            return EmbeddingStats(
                total_entities=0,
                successful_embeddings=0,
                failed_embeddings=0,
                skipped_embeddings=0,
                average_embedding_time=0.0,
                total_time=total_time,
                model_used=self.embedding_model,
                dimension=self.embedding_dimension,
                success=False,
                error_message=str(exc),
            )

    async def close(self):
        """Close connections and cleanup resources."""
        if self.backend:
            await self.backend.close()


__all__ = ["KGEmbedder"]
