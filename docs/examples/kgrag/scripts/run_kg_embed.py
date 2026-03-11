#!/usr/bin/env python3
"""Comprehensive Knowledge Graph Embedding Script.

Generates and stores embeddings for all graph components:
- Entity nodes (Movie, Person, Genre)
- Relations (ACTED_IN, DIRECTED, BELONGS_TO_GENRE)
- Stores embeddings back to Neo4j with vector indices

Follows mellea-contribs architecture using GraphBackend and KGEmbedder.

Usage:
    python run_kg_embed.py --neo4j-uri bolt://localhost:7687
    python run_kg_embed.py --mock  # Mock backend (no embedding)
    python run_kg_embed.py --batch-size 100 --model text-embedding-3-large
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from mellea_contribs.kg.embedder import KGEmbedder
from mellea_contribs.kg.embed_models import EmbeddingConfig, EmbeddingStats
from mellea_contribs.kg.graph_dbs.base import GraphBackend
from mellea_contribs.kg.models import Entity, Relation
from mellea_contribs.kg.utils import (
    create_session,
    create_backend,
    log_progress,
)


@dataclass
class ComprehensiveEmbeddingStats:
    """Statistics for comprehensive embedding operations."""

    start_time: datetime
    end_time: datetime
    duration_seconds: float
    model_used: str
    dimension: int

    # Entity stats
    entities_queried: int
    entities_embedded: int
    entities_failed: int

    # Relation stats
    relations_queried: int
    relations_embedded: int
    relations_failed: int

    # Neo4j storage stats
    entities_stored: int
    relations_stored: int
    vector_indices_created: int

    success: bool
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "model_used": self.model_used,
            "dimension": self.dimension,
            "entities_queried": self.entities_queried,
            "entities_embedded": self.entities_embedded,
            "entities_failed": self.entities_failed,
            "relations_queried": self.relations_queried,
            "relations_embedded": self.relations_embedded,
            "relations_failed": self.relations_failed,
            "entities_stored": self.entities_stored,
            "relations_stored": self.relations_stored,
            "vector_indices_created": self.vector_indices_created,
            "success": self.success,
            "error_message": self.error_message,
        }

    def __str__(self) -> str:
        """Format statistics for display."""
        status = "✓ SUCCESS" if self.success else "✗ FAILED"
        lines = [
            f"Status: {status}",
            f"Duration: {self.duration_seconds:.2f}s",
            f"Model: {self.model_used} (dimension: {self.dimension})",
            "",
            "Entities:",
            f"  Queried: {self.entities_queried}",
            f"  Embedded: {self.entities_embedded}",
            f"  Failed: {self.entities_failed}",
            "",
            "Relations:",
            f"  Queried: {self.relations_queried}",
            f"  Embedded: {self.relations_embedded}",
            f"  Failed: {self.relations_failed}",
            "",
            "Storage:",
            f"  Entities stored: {self.entities_stored}",
            f"  Relations stored: {self.relations_stored}",
            f"  Vector indices created: {self.vector_indices_created}",
        ]
        if self.error_message:
            lines.append(f"Error: {self.error_message}")
        return "\n".join(lines)


class ComprehensiveKGEmbedder:
    """Comprehensive KG embedder using mellea-contribs architecture.

    Embeds entities, relations, and stores results in Neo4j with vector indices.
    Uses KGEmbedder for embedding generation and GraphBackend for persistence.
    """

    def __init__(
        self,
        backend: GraphBackend,
        session: Any,
        embedder: KGEmbedder,
        batch_size: int = 100,
    ):
        """Initialize comprehensive embedder.

        Args:
            backend: GraphBackend for persistence
            session: Mellea session for LLM operations
            embedder: KGEmbedder for generating embeddings
            batch_size: Batch size for processing
        """
        self.backend = backend
        self.session = session
        self.embedder = embedder
        self.batch_size = batch_size

    async def embed_all(self) -> ComprehensiveEmbeddingStats:
        """Run the complete embedding pipeline.

        Returns:
            Statistics about the embedding operation
        """
        start_time = datetime.now()

        try:
            log_progress("=" * 70)
            log_progress("COMPREHENSIVE KG EMBEDDING PIPELINE")
            log_progress("=" * 70)

            # Initialize stats
            entities_embedded = 0
            entities_failed = 0
            entities_stored = 0

            relations_embedded = 0
            relations_failed = 0
            relations_stored = 0

            vector_indices_created = 0

            # Step 1: Embed all entities from Neo4j
            log_progress("\nStep 1: Embedding entities...")
            entities = await self._fetch_entities_from_neo4j()
            log_progress(f"  Fetched {len(entities)} entities from Neo4j")

            embedded_entities, failed = await self._embed_batch(entities)
            entities_embedded = len(embedded_entities)
            entities_failed = failed

            # Store embeddings in Neo4j
            if embedded_entities:
                stored = await self._store_entity_embeddings(embedded_entities)
                entities_stored = stored
                log_progress(f"  ✓ Stored {stored} entity embeddings in Neo4j")

            # Step 2: Embed all relations from Neo4j
            log_progress("\nStep 2: Embedding relations...")
            relations = await self._fetch_relations_from_neo4j()
            log_progress(f"  Fetched {len(relations)} relations from Neo4j")

            embedded_relations, failed = await self._embed_batch(relations)
            relations_embedded = len(embedded_relations)
            relations_failed = failed

            # Store relation embeddings in Neo4j
            if embedded_relations:
                stored = await self._store_relation_embeddings(embedded_relations)
                relations_stored = stored
                log_progress(f"  ✓ Stored {stored} relation embeddings in Neo4j")

            # Step 3: Create vector indices
            log_progress("\nStep 3: Creating vector indices...")
            if self.backend.backend_id == "neo4j":
                indices = await self._create_vector_indices()
                vector_indices_created = indices
                log_progress(f"  ✓ Created {indices} vector indices")

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return ComprehensiveEmbeddingStats(
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                model_used=self.embedder.embedding_model,
                dimension=self.embedder.embedding_dimension,
                entities_queried=len(entities),
                entities_embedded=entities_embedded,
                entities_failed=entities_failed,
                relations_queried=len(relations),
                relations_embedded=relations_embedded,
                relations_failed=relations_failed,
                entities_stored=entities_stored,
                relations_stored=relations_stored,
                vector_indices_created=vector_indices_created,
                success=True,
            )

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            log_progress(f"✗ Embedding failed: {e}")

            return ComprehensiveEmbeddingStats(
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                model_used=self.embedder.embedding_model,
                dimension=self.embedder.embedding_dimension,
                entities_queried=0,
                entities_embedded=0,
                entities_failed=0,
                relations_queried=0,
                relations_embedded=0,
                relations_failed=0,
                entities_stored=0,
                relations_stored=0,
                vector_indices_created=0,
                success=False,
                error_message=str(e),
            )

    async def _fetch_entities_from_neo4j(self) -> List[Entity]:
        """Fetch all entities from Neo4j using Cypher query.

        Returns:
            List of Entity objects from the graph
        """
        if self.backend.backend_id != "neo4j":
            return []

        try:
            # Query all nodes and their labels/properties
            cypher_query = """
            MATCH (n)
            RETURN n.name as name, labels(n)[0] as type, id(n) as node_id
            LIMIT 100000
            """

            if hasattr(self.backend, '_async_driver') and self.backend._async_driver:
                async with self.backend._async_driver.session() as session:
                    result = await session.run(cypher_query)
                    records = [record async for record in result]

                    entities = []
                    for record in records:
                        entity = Entity(
                            type=record.get('type', 'Unknown'),
                            name=record.get('name', ''),
                            description=f"Node of type {record.get('type')}",
                        )
                        entities.append(entity)

                    return entities
        except Exception as e:
            log_progress(f"  Warning: Failed to fetch entities from Neo4j: {e}")
            return []

        return []

    async def _fetch_relations_from_neo4j(self) -> List[Relation]:
        """Fetch all relations from Neo4j using Cypher query.

        Returns:
            List of Relation objects from the graph
        """
        if self.backend.backend_id != "neo4j":
            return []

        try:
            # Query all relationships
            cypher_query = """
            MATCH ()-[r]->()
            RETURN type(r) as relation_type, id(r) as rel_id
            LIMIT 100000
            """

            if hasattr(self.backend, '_async_driver') and self.backend._async_driver:
                async with self.backend._async_driver.session() as session:
                    result = await session.run(cypher_query)
                    records = [record async for record in result]

                    relations = []
                    for record in records:
                        relation = Relation(
                            relation_type=record.get('relation_type', 'UNKNOWN'),
                            source_entity_type='Node',
                            source_entity_name=f"Relation_{record.get('rel_id')}",
                            target_entity_type='Node',
                            target_entity_name='Target',
                        )
                        relations.append(relation)

                    return relations
        except Exception as e:
            log_progress(f"  Warning: Failed to fetch relations from Neo4j: {e}")
            return []

        return []

    async def _embed_batch(
        self,
        items: List[Any]
    ) -> tuple[List[Any], int]:
        """Embed a batch of entities or relations.

        Args:
            items: List of Entity or Relation objects

        Returns:
            Tuple of (successfully_embedded_items, number_failed)
        """
        embedded = []
        failed = 0

        for i, item in enumerate(items):
            try:
                if isinstance(item, Entity):
                    embedded_item = await self.embedder.embed_entity(item)
                    embedded.append(embedded_item)
                elif isinstance(item, Relation):
                    # For relations, embed the relation type name
                    text = f"Relation: {item.relation_type}"
                    embedding = await self.embedder._get_embedding(text)
                    item.embedding = embedding
                    embedded.append(item)

                if (i + 1) % self.batch_size == 0:
                    log_progress(f"    Embedded {i + 1}/{len(items)} items...")

            except Exception as e:
                log_progress(f"    Warning: Failed to embed item {i}: {e}")
                failed += 1
                continue

        return embedded, failed

    async def _store_entity_embeddings(self, entities: List[Entity]) -> int:
        """Store entity embeddings in Neo4j.

        Args:
            entities: List of entities with embeddings

        Returns:
            Number of embeddings stored
        """
        if self.backend.backend_id != "neo4j":
            return 0

        count = 0
        try:
            # Store embeddings back to Neo4j
            cypher_query = """
            UNWIND $batch AS item
            MATCH (n {name: item.name})
            SET n.embedding = item.embedding
            RETURN count(n) as updated
            """

            if hasattr(self.backend, '_async_driver') and self.backend._async_driver:
                batch_data = [
                    {
                        "name": entity.name,
                        "embedding": entity.embedding if hasattr(entity, 'embedding') else [],
                    }
                    for entity in entities
                ]

                async with self.backend._async_driver.session() as session:
                    result = await session.run(cypher_query, batch=batch_data)
                    record = await result.single()
                    if record:
                        count = record.get('updated', 0)
        except Exception as e:
            log_progress(f"  Warning: Failed to store entity embeddings: {e}")

        return count

    async def _store_relation_embeddings(self, relations: List[Relation]) -> int:
        """Store relation embeddings in Neo4j.

        Args:
            relations: List of relations with embeddings

        Returns:
            Number of embeddings stored
        """
        if self.backend.backend_id != "neo4j":
            return 0

        count = 0
        try:
            # Store relation embeddings
            cypher_query = """
            UNWIND $batch AS item
            MATCH ()-[r {type: item.relation_type}]->()
            SET r.embedding = item.embedding
            RETURN count(r) as updated
            """

            if hasattr(self.backend, '_async_driver') and self.backend._async_driver:
                batch_data = [
                    {
                        "relation_type": relation.relation_type,
                        "embedding": relation.embedding if hasattr(relation, 'embedding') else [],
                    }
                    for relation in relations
                ]

                async with self.backend._async_driver.session() as session:
                    result = await session.run(cypher_query, batch=batch_data)
                    record = await result.single()
                    if record:
                        count = record.get('updated', 0)
        except Exception as e:
            log_progress(f"  Warning: Failed to store relation embeddings: {e}")

        return count

    async def _create_vector_indices(self) -> int:
        """Create Neo4j vector indices for embedding search.

        Returns:
            Number of indices created
        """
        if self.backend.backend_id != "neo4j":
            return 0

        indices_created = 0
        try:
            if hasattr(self.backend, '_async_driver') and self.backend._async_driver:
                async with self.backend._async_driver.session() as session:
                    # Create entity vector index
                    entity_index_query = f"""
                    CREATE VECTOR INDEX IF NOT EXISTS entity_embedding_index
                    FOR (n) ON (n.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {self.embedder.embedding_dimension},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                    """
                    try:
                        await session.run(entity_index_query)
                        indices_created += 1
                        log_progress("    Created entity embedding index")
                    except Exception as e:
                        log_progress(f"    Note: Entity index creation: {e}")

                    # Create relation vector index
                    relation_index_query = f"""
                    CREATE VECTOR INDEX IF NOT EXISTS relation_embedding_index
                    FOR (r: RELATIONSHIP) ON (r.embedding)
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {self.embedder.embedding_dimension},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                    """
                    try:
                        await session.run(relation_index_query)
                        indices_created += 1
                        log_progress("    Created relation embedding index")
                    except Exception as e:
                        log_progress(f"    Note: Relation index creation: {e}")

        except Exception as e:
            log_progress(f"  Warning: Failed to create vector indices: {e}")

        return indices_created


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive KG embedding with Neo4j storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --neo4j-uri bolt://localhost:7687
  %(prog)s --batch-size 500 --model text-embedding-3-large
  %(prog)s --mock  # Mock backend (no actual embedding)
        """,
    )

    # Backend configuration
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)",
    )

    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Neo4j username (default: neo4j)",
    )

    parser.add_argument(
        "--neo4j-password",
        type=str,
        default="password",
        help="Neo4j password (default: password)",
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockGraphBackend (no Neo4j needed)",
    )

    # Embedding configuration
    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model (default: text-embedding-3-small)",
    )

    parser.add_argument(
        "--dimension",
        type=int,
        default=1536,
        help="Embedding dimension (default: 1536)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    try:
        # Initialize backend and session
        backend = create_backend(
            backend_type="neo4j" if not args.mock else "mock",
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
        )
        session = create_session(model_id="gpt-4o-mini")

        # Create embedder
        config = EmbeddingConfig(
            model=args.model,
            dimension=args.dimension,
            batch_size=args.batch_size,
        )

        embedder = KGEmbedder(
            session=session,
            model=args.model,
            dimension=args.dimension,
            batch_size=args.batch_size,
            backend=backend,
        )

        # Run comprehensive embedding
        log_progress("")
        comprehensive_embedder = ComprehensiveKGEmbedder(
            backend=backend,
            session=session,
            embedder=embedder,
            batch_size=args.batch_size,
        )

        stats = await comprehensive_embedder.embed_all()

        # Print results
        log_progress("")
        log_progress("=" * 70)
        log_progress("EMBEDDING STATISTICS")
        log_progress("=" * 70)
        log_progress(str(stats))
        log_progress("=" * 70)
        log_progress("")

        # Output JSON
        print(json.dumps(stats.to_dict()))

        # Return appropriate exit code
        sys.exit(0 if stats.success else 1)

    except KeyboardInterrupt:
        log_progress("\n⚠️  Embedding interrupted by user")
        sys.exit(130)
    except Exception as e:
        log_progress(f"❌ Embedding failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
