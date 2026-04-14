#!/usr/bin/env python3
"""Knowledge Graph Preprocessing from Predefined Data.

Loads movie and person databases and inserts them into the graph database using
:class:`~preprocessor.movie_preprocessor.PredefinedDataPreprocessor`.

Usage:
    python run_kg_preprocess.py --data-dir ./dataset/movie --db-uri bolt://localhost:7687
    python run_kg_preprocess.py --data-dir ./dataset/movie --mock
    python run_kg_preprocess.py --data-dir ./dataset/movie --verbose
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from mellea_contribs.kg.utils import (
    create_backend,
    log_progress,
)

from preprocessor.movie_preprocessor import PredefinedDataPreprocessor


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess and load predefined movie data into KG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --data-dir ./data/movie                                # Load from data directory
  %(prog)s --data-dir ./data/movie --mock                         # Use mock backend
  %(prog)s --data-dir ./data/movie --db-uri bolt://localhost:7687 # Custom graph DB URI
  %(prog)s --data-dir ./data/movie --verbose                      # Verbose logging
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing movie_db.json and person_db.json",
    )
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
        help="Use MockGraphBackend instead of the graph database (no database needed)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for inserting entities (default: 50)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log_progress(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    for filename in ["movie_db.json", "person_db.json"]:
        if not (data_dir / filename).exists():
            log_progress(f"ERROR: Required file not found: {data_dir / filename}")
            sys.exit(1)

    try:
        backend = create_backend(
            backend_type="neo4j" if not args.mock else "mock",
            neo4j_uri=args.db_uri,
            neo4j_user=args.db_user,
            neo4j_password=args.db_password,
        )
        log_progress("=" * 60)
        log_progress("KG Preprocessing from Predefined Data")
        log_progress("=" * 60)
        log_progress(f"Data directory: {data_dir}")
        log_progress(f"Backend: {'Mock' if args.mock else 'Graph DB'}")
        log_progress("")

        preprocessor = PredefinedDataPreprocessor(
            backend=backend,
            data_dir=data_dir,
            batch_size=args.batch_size,
        )

        stats = await preprocessor.preprocess()

        log_progress("")
        log_progress("=" * 60)
        log_progress("PREPROCESSING SUMMARY")
        log_progress("=" * 60)
        log_progress(str(stats))
        log_progress("=" * 60)
        log_progress("")

        print(json.dumps(stats.to_dict()))
        sys.exit(0 if stats.success else 1)

    except KeyboardInterrupt:
        log_progress("\n⚠️  Preprocessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        log_progress(f"❌ Preprocessing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
