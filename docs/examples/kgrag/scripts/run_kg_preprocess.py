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
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path for relative imports
_SCRIPT_DIR = Path(__file__).parent
_EXAMPLES_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_EXAMPLES_DIR))

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

from mellea_contribs.kg.updater_models import UpdateStats
from preprocessor.movie_preprocessor import MovieKGPreprocessor


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


async def process_documents(
    input_path: Path,
    backend: GraphBackend,
    session: MelleaSession,
    model: str,
    batch_size: int,
) -> UpdateStats:
    """Process documents and build KG."""
    preprocessor = MovieKGPreprocessor(
        backend=backend,
        session=session,
        batch_size=batch_size,
    )

    stats = UpdateStats()
    start_time = time.time()

    try:
        with open(input_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    doc = json.loads(line)
                    doc_id = doc.get("id", f"doc_{line_num}")
                    doc_text = doc.get("text", "")

                    if not doc_text:
                        print(f"[{line_num}] WARNING: Empty text for {doc_id}", file=sys.stderr)
                        continue

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

                    print(
                        f"[{line_num}] Processed {doc_id}: "
                        f"{len(result.entities)} entities, {len(result.relations)} relations "
                        f"({doc_elapsed:.2f}s)",
                        file=sys.stderr
                    )

                except json.JSONDecodeError as e:
                    print(f"[{line_num}] ERROR: Failed to parse JSON: {e}", file=sys.stderr)
                    stats.total_documents += 1
                    stats.failed_documents += 1
                except Exception as e:
                    print(f"[{line_num}] ERROR: {e}", file=sys.stderr)
                    stats.total_documents += 1
                    stats.failed_documents += 1

    except IOError as e:
        print(f"ERROR: Failed to read input file: {e}", file=sys.stderr)
        return stats

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
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Initialize backend and session
    backend = build_backend(args)
    session = start_session(backend_name="litellm", model_id=args.model)

    try:
        # Process documents
        stats = await process_documents(
            input_path=input_path,
            backend=backend,
            session=session,
            model=args.model,
            batch_size=args.batch_size,
        )

        # Print stats
        print("\n=== KG Preprocessing Stats ===", file=sys.stderr)
        print(f"Total documents: {stats.total_documents}", file=sys.stderr)
        print(f"Successful: {stats.successful_documents}", file=sys.stderr)
        print(f"Failed: {stats.failed_documents}", file=sys.stderr)
        print(f"Entities extracted: {stats.entities_extracted}", file=sys.stderr)
        print(f"Relations extracted: {stats.relations_extracted}", file=sys.stderr)
        print(f"Entities added: {stats.entities_new}", file=sys.stderr)
        print(f"Relations added: {stats.relations_new}", file=sys.stderr)
        print(
            f"Average time per doc: {stats.average_processing_time_per_doc_ms:.2f}ms",
            file=sys.stderr
        )

        # Output stats to file if requested
        if args.output_stats:
            output_path = Path(args.output_stats)
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
