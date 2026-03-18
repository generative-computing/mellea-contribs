#!/usr/bin/env python3
"""Knowledge Graph Embedding Script.

Generates and stores embeddings for all graph components:
- Entity nodes (Movie, Person, Genre)
- Relations (ACTED_IN, DIRECTED, BELONGS_TO_GENRE)
- Stores embeddings back to the graph database with vector indices

Delegates to :class:`~mellea_contribs.kg.embedder.KGEmbedder` which handles
fetch / embed / store / create-index in a single
:meth:`~mellea_contribs.kg.embedder.KGEmbedder.embed_and_store_all` call.

Usage:
    python run_kg_embed.py --db-uri bolt://localhost:7687
    python run_kg_embed.py --mock  # Mock backend (no actual embedding)
    python run_kg_embed.py --batch-size 100 --model text-embedding-3-large
"""

import argparse
import asyncio
import json
import os
import sys

from dotenv import load_dotenv

from mellea_contribs.kg.embedder import KGEmbedder
from mellea_contribs.kg.utils import (
    create_backend,
    create_session_from_env,
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
  %(prog)s --db-uri bolt://localhost:7687
  %(prog)s --batch-size 500 --model text-embedding-3-large
  %(prog)s --mock  # Mock backend (no actual embedding)
        """,
    )

    # Backend configuration
    parser.add_argument(
        "--db-uri",
        type=str,
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Graph database connection URI (default: $NEO4J_URI or bolt://localhost:7687)",
    )
    parser.add_argument(
        "--db-user",
        type=str,
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Graph database username (default: $NEO4J_USER or neo4j)",
    )
    parser.add_argument(
        "--db-password",
        type=str,
        default=os.getenv("NEO4J_PASSWORD", "password"),
        help="Graph database password (default: $NEO4J_PASSWORD or password)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockGraphBackend (no graph database needed)",
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

    # Load .env from the parent directory (docs/examples/kgrag/.env)
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(env_path, override=False)

    backend = create_backend(
        backend_type="neo4j" if not args.mock else "mock",
        neo4j_uri=args.db_uri,
        neo4j_user=args.db_user,
        neo4j_password=args.db_password,
    )
    session, _ = create_session_from_env()

    emb_api_base = os.getenv("EMB_API_BASE")
    emb_api_key = os.getenv("EMB_API_KEY")
    emb_model = os.getenv("EMB_MODEL_NAME", args.model)
    emb_dimension = int(os.getenv("VECTOR_DIMENSIONS", str(args.dimension)))
    # RITS authenticates via a custom header; fall back to the primary key.
    rits_api_key = os.getenv("EMB_RITS_API_KEY") or os.getenv("RITS_API_KEY")
    extra_headers = {"RITS_API_KEY": rits_api_key} if rits_api_key else {}

    embedder = KGEmbedder(
        session=session,
        model=emb_model,
        dimension=emb_dimension,
        api_base=emb_api_base,
        api_key=emb_api_key,
        extra_headers=extra_headers,
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
