#!/usr/bin/env python3
"""Generate embeddings for entities in the KG via KGEmbedder.

Reads entities from the graph backend and generates embeddings using
the KGEmbedder with a configurable embedding model.

Example::

    python run_kg_embed.py --mock --model text-embedding-3-small
    python run_kg_embed.py --neo4j-uri bolt://localhost:7687 --model text-embedding-3-small
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Optional

try:
    from mellea import start_session, MelleaSession
except ImportError:
    print("Error: mellea not installed. Run: pip install mellea[litellm]")
    sys.exit(1)

from mellea_contribs.kg.graph_dbs.base import GraphBackend
from mellea_contribs.kg.graph_dbs.mock import MockGraphBackend

try:
    from mellea_contribs.kg.graph_dbs.neo4j import Neo4jBackend
except ImportError:
    Neo4jBackend = None

from mellea_contribs.kg.embedder import KGEmbedder
from mellea_contribs.kg.embed_models import EmbeddingConfig, EmbeddingStats


def build_backend(args) -> GraphBackend:
    """Build graph backend from command line arguments."""
    if args.mock:
        return MockGraphBackend()
    if Neo4jBackend is None:
        print("Error: Neo4j backend not available. Install: pip install mellea-contribs[kg]")
        sys.exit(1)
    return Neo4jBackend(
        connection_uri=args.neo4j_uri,
        auth=(args.neo4j_user, args.neo4j_password),
    )


async def embed_entities(
    backend: GraphBackend,
    model: str,
    dimension: int,
    batch_size: int,
) -> EmbeddingStats:
    """Embed all entities in the KG."""
    embedder = KGEmbedder(backend=backend)

    config = EmbeddingConfig(
        model=model,
        dimension=dimension,
        batch_size=batch_size,
    )

    start_time = time.time()

    try:
        # Fetch entities from backend (mock backend returns empty list by default)
        entities = await backend.get_all_nodes()
        print(f"Found {len(entities)} entities to embed", file=sys.stderr)

        if not entities:
            # For mock backend, return zero stats
            stats = EmbeddingStats(
                total_entities=0,
                successful_embeddings=0,
                failed_embeddings=0,
                skipped_embeddings=0,
                average_embedding_time=0.0,
                total_time=0.0,
                model_used=model,
            )
            return stats

        # Embed entities
        # Note: KGEmbedder.embed_batch expects text representations
        # For now, we'll use entity names or simple text
        entity_texts = [
            getattr(e, "name", f"Entity_{i}") for i, e in enumerate(entities)
        ]

        embeddings = await embedder.embed_batch(
            texts=entity_texts,
            config=config,
        )

        elapsed = time.time() - start_time

        # Build stats
        stats = EmbeddingStats(
            total_entities=len(entities),
            successful_embeddings=len(embeddings),
            failed_embeddings=0,
            skipped_embeddings=0,
            average_embedding_time=elapsed / len(embeddings) if embeddings else 0.0,
            total_time=elapsed,
            model_used=model,
        )

        return stats

    except Exception as e:
        print(f"ERROR: Failed to embed entities: {e}", file=sys.stderr)
        elapsed = time.time() - start_time
        stats = EmbeddingStats(
            total_entities=0,
            successful_embeddings=0,
            failed_embeddings=1,
            skipped_embeddings=0,
            average_embedding_time=0.0,
            total_time=elapsed,
            model_used=model,
        )
        return stats


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for entities in the KG"
    )
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
        help="Use MockGraphBackend instead of Neo4j (no database needed)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model to use (default: text-embedding-3-small)",
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
        default=10,
        help="Batch size for embedding (default: 10)",
    )
    parser.add_argument(
        "--output-stats",
        type=str,
        help="Output JSON file for EmbeddingStats (optional)",
    )
    args = parser.parse_args()

    # Initialize backend and session
    backend = build_backend(args)
    session = start_session(backend_name="litellm", model_id="gpt-4o-mini")

    try:
        # Embed entities
        stats = await embed_entities(
            backend=backend,
            model=args.model,
            dimension=args.dimension,
            batch_size=args.batch_size,
        )

        # Print stats
        print("\n=== Embedding Stats ===", file=sys.stderr)
        print(f"Total entities: {stats.total_entities}", file=sys.stderr)
        print(f"Successful: {stats.successful_embeddings}", file=sys.stderr)
        print(f"Failed: {stats.failed_embeddings}", file=sys.stderr)
        print(f"Skipped: {stats.skipped_embeddings}", file=sys.stderr)
        print(f"Average time per entity: {stats.average_embedding_time:.4f}s", file=sys.stderr)
        print(f"Total time: {stats.total_time:.2f}s", file=sys.stderr)
        print(f"Model: {stats.model_used}", file=sys.stderr)

        # Output stats to file if requested
        if args.output_stats:
            import pathlib
            output_path = pathlib.Path(args.output_stats)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(stats.model_dump(), f, indent=2)
            print(f"\nStats saved to: {output_path}", file=sys.stderr)

        # Write stats to stdout as JSON
        print(json.dumps(stats.model_dump()))

    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
