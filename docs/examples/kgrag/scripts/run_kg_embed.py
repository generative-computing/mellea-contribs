#!/usr/bin/env python3
"""Knowledge Graph Embedding Script.

Generates and stores embeddings for all graph components:
- Entity nodes (Movie, Person, Genre)
- Relations (ACTED_IN, DIRECTED, BELONGS_TO_GENRE)
- Stores embeddings back to Neo4j with vector indices

Delegates to :class:`~mellea_contribs.kg.embedder.KGEmbedder` which handles
fetch / embed / store / create-index in a single
:meth:`~mellea_contribs.kg.embedder.KGEmbedder.embed_and_store_all` call.

Usage:
    python run_kg_embed.py --neo4j-uri bolt://localhost:7687
    python run_kg_embed.py --mock  # Mock backend (no actual embedding)
    python run_kg_embed.py --batch-size 100 --model text-embedding-3-large
"""

import argparse
import asyncio
import json
import sys

from mellea_contribs.kg.embedder import KGEmbedder
from mellea_contribs.kg.utils import (
    create_backend,
    create_session,
    log_progress,
    output_json,
    print_stats,
)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Embed all KG entities/relations and store them in the graph",
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
        help="Batch size for progress logging (default: 100)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    backend = create_backend(
        backend_type="neo4j" if not args.mock else "mock",
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
    )
    session = create_session(model_id="gpt-4o-mini")

    embedder = KGEmbedder(
        session=session,
        model=args.model,
        dimension=args.dimension,
        batch_size=args.batch_size,
        backend=backend,
    )

    try:
        log_progress("=" * 60)
        log_progress("KG Embedding Pipeline")
        log_progress("=" * 60)

        stats = await embedder.embed_and_store_all(batch_size=args.batch_size)

        log_progress("=" * 60)
        log_progress("EMBEDDING SUMMARY")
        log_progress("=" * 60)
        print_stats(stats)
        log_progress("=" * 60)

        output_json(stats)
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
