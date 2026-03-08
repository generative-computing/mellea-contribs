#!/usr/bin/env python3
"""Preprocess documents and build KG via MovieKGPreprocessor.

Reads documents from a JSONL file, extracts entities and relations using
the domain-specific MovieKGPreprocessor, and persists to the KG backend.

Input format: Each line is JSON with 'id' and 'text' fields.
Output: Per-document progress and final UpdateStats.

Example::

    python run_kg_preprocess.py --input data/documents.jsonl --mock
    python run_kg_preprocess.py --input data/documents.jsonl --neo4j-uri bolt://localhost:7687
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path for relative imports
_SCRIPT_DIR = Path(__file__).parent
_EXAMPLES_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_EXAMPLES_DIR))

from mellea_contribs.kg.updater_models import UpdateStats
from mellea_contribs.kg.utils import (
    create_session,
    create_backend,
    load_jsonl,
    log_progress,
    output_json,
    print_stats,
)
from preprocessor.movie_preprocessor import MovieKGPreprocessor


async def process_documents(
    input_path: Path,
    backend,
    session,
    model: str,
    batch_size: int,
) -> UpdateStats:
    """Process documents and build KG."""
    from mellea_contribs.kg.graph_dbs.base import GraphBackend

    preprocessor = MovieKGPreprocessor(
        backend=backend,
        session=session,
        batch_size=batch_size,
    )

    stats = UpdateStats()
    start_time = time.time()

    try:
        for line_num, doc in enumerate(load_jsonl(input_path), 1):
            doc_id = doc.get("id", f"doc_{line_num}")
            doc_text = doc.get("text", "")

            if not doc_text:
                log_progress(f"[{line_num}] WARNING: Empty text for {doc_id}")
                continue

            try:
                # Process document
                doc_start = time.time()
                result = await preprocessor.process_document(
                    doc_text=doc_text,
                    doc_id=doc_id,
                )
                doc_elapsed = time.time() - doc_start

                # Update stats
                stats.total_documents += 1
                stats.successful_documents += 1
                stats.entities_extracted += len(result.entities)
                stats.relations_extracted += len(result.relations)

                # Simple persistence tracking
                stats.entities_new += len(result.entities)
                stats.relations_new += len(result.relations)

                log_progress(
                    f"[{line_num}] Processed {doc_id}: "
                    f"{len(result.entities)} entities, {len(result.relations)} relations "
                    f"({doc_elapsed:.2f}s)"
                )

            except Exception as e:
                log_progress(f"[{line_num}] ERROR: {e}")
                stats.total_documents += 1
                stats.failed_documents += 1

    except Exception as e:
        log_progress(f"ERROR: Failed to read input file: {e}")

    elapsed = time.time() - start_time
    stats.total_processing_time_ms = elapsed * 1000

    if stats.successful_documents > 0:
        stats.average_processing_time_per_doc_ms = (
            stats.total_processing_time_ms / stats.successful_documents
        )

    return stats


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess documents and build KG via MovieKGPreprocessor"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file path (each line: {\"id\": \"...\", \"text\": \"...\"})",
    )
    parser.add_argument(
        "--output-stats",
        type=str,
        help="Output JSON file for UpdateStats (optional)",
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
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for parallel processing (default: 10)",
    )
    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        log_progress(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Initialize backend and session using utilities
    backend = create_backend(
        backend_type="neo4j" if not args.mock else "mock",
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
    )
    session = create_session(model_id=args.model)

    try:
        # Process documents
        stats = await process_documents(
            input_path=input_path,
            backend=backend,
            session=session,
            model=args.model,
            batch_size=args.batch_size,
        )

        # Print stats using utility
        log_progress("\n=== KG Preprocessing Stats ===")
        print_stats(stats, to_stderr=True)

        # Output stats to file if requested
        if args.output_stats:
            import json

            output_path = Path(args.output_stats)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(stats.model_dump(), f, indent=2)
            log_progress(f"Stats saved to: {output_path}")

        # Write stats to stdout as JSON
        output_json(stats)

    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
