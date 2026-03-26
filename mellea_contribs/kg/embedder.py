"""KG Embedder: Layer 2 library for generating vector embeddings for KG entities.

This module provides embedding infrastructure for converting entities and relations
into vector representations using LiteLLM's embedding API.

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
import asyncio
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

    Orchestrates embedding operations using LiteLLM's embedding API.

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
        extra_headers: Optional[dict] = None,
        batch_size: int = 10,
        backend: Optional[GraphBackend] = None,
    ):
        """Initialize the KG embedder.

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
            api_key: Optional API key for embedding service. When api_base is
                set and no key is provided, defaults to "dummy" so that
                OpenAI-compatible endpoints that authenticate via custom
                headers (e.g. RITS) do not raise an auth error.
            extra_headers: Optional dict of extra HTTP headers forwarded to
                the embedding endpoint (e.g. {"RITS_API_KEY": "..."}).
            batch_size: Number of entities to embed in parallel per batch.
                Default: 10
            backend: Optional GraphBackend for persisting embeddings
        """

        self.session = session
        # When routing through a custom OpenAI-compatible endpoint (api_base
        # is set) and the model name doesn't already carry a LiteLLM provider
        # prefix (e.g. "openai/", "huggingface/", "azure/"), prepend "openai/"
        # so LiteLLM routes to the correct adapter instead of raising
        # "LLM Provider NOT provided".
        _LITELLM_PROVIDERS = {
            "openai", "azure", "huggingface", "ollama", "cohere",
            "anthropic", "replicate", "together_ai", "vertex_ai",
        }
        if api_base and model.split("/")[0] not in _LITELLM_PROVIDERS:
            model = f"openai/{model}"
        self.embedding_model = model
        self.embedding_dimension = dimension
        self.api_base = api_base
        # Default to "dummy" when a custom endpoint is in use — services like
        # RITS authenticate via extra_headers rather than a bearer token.
        self.api_key = api_key or ("dummy" if api_base else None)
        self.extra_headers = extra_headers or {}
        self.batch_size = batch_size
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

            kwargs: dict = {
                "model": self.embedding_model,
                "input": text,
                "encoding_format": "float",
            }
            if self.api_base:
                kwargs["api_base"] = self.api_base
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.extra_headers:
                kwargs["extra_headers"] = self.extra_headers
            response = await litellm.aembedding(**kwargs)

            # LiteLLM returns an EmbeddingResponse object with a .data list.
            # Items may be Embedding objects (.embedding) or plain dicts.
            if hasattr(response, "data"):
                item = response.data[0]
                return item["embedding"] if isinstance(item, dict) else item.embedding
            # Fallback for plain dict responses
            if isinstance(response, dict) and "data" in response:
                return response["data"][0]["embedding"]
            # Fallback for plain list responses
            if isinstance(response, list):
                return response[0]["embedding"]
            return response
        except Exception as e:
            logger.error(f"Embedding API error: {e}")
            raise

    async def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in a single API call.

        OpenAI-compatible endpoints accept a list as the ``input`` field,
        which is far more efficient than one call per text.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            Exception: If the embedding API call fails.
        """
        import litellm

        kwargs: dict = {
            "model": self.embedding_model,
            "input": texts,
            "encoding_format": "float",
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers

        response = await litellm.aembedding(**kwargs)

        if hasattr(response, "data"):
            items = response.data
            return [
                item["embedding"] if isinstance(item, dict) else item.embedding
                for item in items
            ]
        if isinstance(response, dict) and "data" in response:
            return [item["embedding"] for item in response["data"]]
        raise ValueError(f"Unexpected embedding response format: {type(response)}")

    # ------------------------------------------------------------------
    # Embedding pipeline: fetch / store / index (delegated to backend)
    # ------------------------------------------------------------------

    async def fetch_entities_for_embedding(self) -> list[Entity]:
        """Fetch all entities from the backend for embedding.

        Returns:
            List of Entity objects with ``id`` populated, or empty list if
            the backend does not support bulk fetch.
        """
        if not self.backend:
            return []
        rows = await self.backend.fetch_entities_for_embedding()
        entities = []
        for r in rows:
            e = Entity(
                type=r.get("type", "Unknown"),
                name=r.get("name", "") or "",
                description=r.get("description", "") or "",
            )
            e.id = r.get("eid", "")
            entities.append(e)
        return entities

    async def fetch_relations_for_embedding(self) -> list[Relation]:
        """Fetch all relations from the backend for embedding.

        Returns:
            List of Relation objects with ``id`` populated, or empty list if
            the backend does not support bulk fetch.
        """
        if not self.backend:
            return []
        rows = await self.backend.fetch_relations_for_embedding()
        relations = []
        for r in rows:
            rel = Relation(
                source_entity=r.get("src_name", "") or "",
                relation_type=r.get("relation_type", "UNKNOWN"),
                target_entity=r.get("dst_name", "") or "",
            )
            rel.id = r.get("eid", "")
            relations.append(rel)
        return relations

    async def store_entity_embeddings(
        self, entities: list[Entity], store_batch_size: int = 50_000
    ) -> int:
        """Store entity embeddings via the backend.

        Args:
            entities: Entities with populated ``embedding`` and ``id`` fields.
            store_batch_size: Items per backend call (default: 50,000).

        Returns:
            Number of embeddings stored.
        """
        if not self.backend:
            return 0
        rows = [
            {"eid": getattr(e, "id", ""), "embedding": getattr(e, "embedding", []) or []}
            for e in entities
            if getattr(e, "embedding", None) and getattr(e, "id", None)
        ]
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None
        chunks = [rows[i : i + store_batch_size] for i in range(0, len(rows), store_batch_size)]
        total_stored = 0
        pbar = _tqdm(total=len(rows), desc="Storing entity embeddings", unit="ent") if _tqdm else None
        try:
            for chunk in chunks:
                total_stored += await self.backend.store_node_embeddings(chunk)
                if pbar:
                    pbar.update(len(chunk))
        finally:
            if pbar:
                pbar.close()
        return total_stored

    async def store_relation_embeddings(
        self, relations: list[Relation], store_batch_size: int = 50_000
    ) -> int:
        """Store relation embeddings via the backend.

        Args:
            relations: Relations with populated ``embedding`` and ``id`` fields.
            store_batch_size: Items per backend call (default: 50,000).

        Returns:
            Number of embeddings stored.
        """
        if not self.backend:
            return 0
        rows = [
            {"eid": getattr(r, "id", ""), "embedding": getattr(r, "embedding", []) or []}
            for r in relations
            if getattr(r, "embedding", None) and getattr(r, "id", None)
        ]
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None
        chunks = [rows[i : i + store_batch_size] for i in range(0, len(rows), store_batch_size)]
        total_stored = 0
        pbar = _tqdm(total=len(rows), desc="Storing relation embeddings", unit="rel") if _tqdm else None
        try:
            for chunk in chunks:
                total_stored += await self.backend.store_edge_embeddings(chunk)
                if pbar:
                    pbar.update(len(chunk))
        finally:
            if pbar:
                pbar.close()
        return total_stored

    async def create_vector_indices(self) -> int:
        """Create vector similarity indices via the backend.

        Returns:
            Number of indices created.
        """
        if not self.backend:
            return 0
        dim = self.embedding_dimension
        indices_created = 0
        for name, target, prop in [
            ("entity_embedding_index", "_Embeddable", "_embedding"),
            ("relation_embedding_index", "RELATIONSHIP", "_embedding"),
        ]:
            if await self.backend.create_vector_index(name, target, prop, dim):
                indices_created += 1
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
            try:
                from tqdm import tqdm as _tqdm
            except ImportError:
                _tqdm = None

            # --- entities ---------------------------------------------------
            logger.info("Embedding pipeline: fetching entities…")
            entities = await self.fetch_entities_for_embedding()
            logger.info(f"  Fetched {len(entities)} entities")

            semaphore = asyncio.Semaphore(64)  # max concurrent batch requests

            async def _embed_entity_batch(batch: list) -> tuple[int, int]:
                """Embed one batch via a single API call; return (ok, failed)."""
                texts = []
                for ent in batch:
                    # Match Bidirection's entity_to_text format: (Type: Name, desc: "...")
                    desc = ent.description or ""
                    if desc:
                        text = f"({ent.type}: {ent.name}, desc: \"{desc}\")"
                    else:
                        text = f"({ent.type}: {ent.name})"
                    texts.append(text)
                async with semaphore:
                    try:
                        embeddings = await self._get_embeddings_batch(texts)
                        for ent, emb in zip(batch, embeddings):
                            ent.embedding = emb
                        return len(batch), 0
                    except Exception as exc:
                        logger.error(f"Batch embed error: {exc}")
                        return 0, len(batch)

            entity_batches = [
                entities[i : i + batch_size]
                for i in range(0, len(entities), batch_size)
            ]
            pbar = _tqdm(total=len(entities), desc="Embedding entities", unit="ent") if _tqdm else None
            tasks = [_embed_entity_batch(b) for b in entity_batches]
            for coro in asyncio.as_completed(tasks):
                ok, failed = await coro
                entities_embedded += ok
                entities_failed += failed
                if pbar:
                    pbar.update(ok + failed)
                else:
                    done = entities_embedded + entities_failed
                    if done % (batch_size * 10) == 0:
                        logger.info(f"  Embedded {done}/{len(entities)} entities…")
            if pbar:
                pbar.close()

            if entities_embedded:
                entities_stored = await self.store_entity_embeddings(entities)
                logger.info(f"  Stored {entities_stored} entity embeddings")

            # --- relations --------------------------------------------------
            logger.info("Embedding pipeline: fetching relations…")
            relations = await self.fetch_relations_for_embedding()
            logger.info(f"  Fetched {len(relations)} relations")

            async def _embed_relation_batch(batch: list) -> tuple[int, int]:
                texts = [f"Relation: {r.relation_type}" for r in batch]
                async with semaphore:
                    try:
                        embeddings = await self._get_embeddings_batch(texts)
                        for rel, emb in zip(batch, embeddings):
                            rel.embedding = emb  # type: ignore[attr-defined]
                        return len(batch), 0
                    except Exception as exc:
                        logger.error(f"Relation batch embed error: {exc}")
                        return 0, len(batch)

            relation_batches = [
                relations[i : i + batch_size]
                for i in range(0, len(relations), batch_size)
            ]
            pbar = _tqdm(total=len(relations), desc="Embedding relations", unit="rel") if _tqdm else None
            tasks = [_embed_relation_batch(b) for b in relation_batches]
            for coro in asyncio.as_completed(tasks):
                ok, failed = await coro
                relations_embedded += ok
                relations_failed += failed
                if pbar:
                    pbar.update(ok + failed)
                else:
                    done = relations_embedded + relations_failed
                    if done % (batch_size * 10) == 0:
                        logger.info(f"  Embedded {done}/{len(relations)} relations…")
            if pbar:
                pbar.close()

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
