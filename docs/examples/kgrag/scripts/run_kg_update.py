#!/usr/bin/env python3
"""Update the KG with new documents via orchestrate_kg_update.

Reads documents from a JSONL file and updates the KG using the orchestration
function. Produces UpdateBatchResult written to output JSONL file.

Input format: Each line is JSON with 'id' and 'text' fields.
Output format: UpdateBatchResult fields as JSON.

Example::

    python run_kg_update.py --input data/new_docs.jsonl --output /tmp/update_results.jsonl --mock
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from mellea_contribs.kg.kgrag import orchestrate_kg_update
from mellea_contribs.kg.updater_models import (
    UpdateBatchResult,
    UpdateResult,
    UpdateStats,
)
from mellea_contribs.kg.utils import (
    create_session,
    create_backend,
    load_jsonl,
    log_progress,
    output_json,
    print_stats,
)


async def update_kg_with_documents(
    input_path: Path,
    backend,
    session,
    model: str,
    domain: str,
    batch_size: int,
) -> UpdateBatchResult:
    """Update KG with documents from JSONL file."""
    batch_result = UpdateBatchResult()
    batch_start = time.time()
    results = []

    try:
        for line_num, doc in enumerate(load_jsonl(input_path), 1):
            doc_id = doc.get("id", f"doc_{line_num}")
            doc_text = doc.get("text", "")

            if not doc_text:
                log_progress(f"[{line_num}] WARNING: Empty text for {doc_id}")
                continue

            doc_start = time.time()

            # Call orchestrate_kg_update
            try:
                update_result = await orchestrate_kg_update(
                    session=session,
                    backend=backend,
                    doc_text=doc_text,
                    domain=domain,
                    hints="",
                    entity_types="",
                    relation_types="",
                )

                doc_elapsed = time.time() - doc_start

                # Create UpdateResult from response
                result = UpdateResult(
                    document_id=doc_id,
                    success=True,
                    entities_found=len(
                        update_result.get("extracted_entities", [])
                    ),
                    relations_found=len(
                        update_result.get("extracted_relations", [])
                    ),
                    entities_added=len(
                        update_result.get("extracted_entities", [])
                    ),
                    relations_added=len(
                        update_result.get("extracted_relations", [])
                    ),
                    processing_time_ms=doc_elapsed * 1000,
                    model_used=model,
                )
                results.append(result)
                batch_result.successful_documents += 1

                log_progress(
                    f"[{line_num}] Updated {doc_id}: "
                    f"{result.entities_found} entities, {result.relations_found} relations "
                    f"({doc_elapsed:.2f}s)"
                )

            except Exception as inner_e:
                log_progress(f"[{line_num}] ERROR in update: {inner_e}")
                result = UpdateResult(
                    document_id=doc_id,
                    success=False,
                    error=str(inner_e),
                    processing_time_ms=(time.time() - doc_start) * 1000,
                )
                results.append(result)
                batch_result.failed_documents += 1

    except Exception as e:
        log_progress(f"ERROR: Failed to read input file: {e}")
        return batch_result

    batch_elapsed = time.time() - batch_start
    batch_result.total_documents = len(results)
    batch_result.results = results
    batch_result.total_time_ms = batch_elapsed * 1000

    if len(results) > 0:
        batch_result.avg_time_per_document_ms = (
            batch_result.total_time_ms / len(results)
        )

        # Aggregate stats
        stats = UpdateStats()
        stats.total_documents = len(results)
        stats.successful_documents = batch_result.successful_documents
        stats.failed_documents = batch_result.failed_documents
        for result in results:
            stats.entities_extracted += result.entities_found
            stats.relations_extracted += result.relations_found
            stats.entities_new += result.entities_added
            stats.relations_new += result.relations_added
        batch_result.stats = stats

    return batch_result


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update KG with documents via orchestrate_kg_update"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSONL file for UpdateBatchResult (optional)",
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
        help="Use MockGraphBackend instead of Neo4j",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="general",
        help="Domain for extraction hints (default: general)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10)",
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
        # Update KG
        batch_result = await update_kg_with_documents(
            input_path=input_path,
            backend=backend,
            session=session,
            model=args.model,
            domain=args.domain,
            batch_size=args.batch_size,
        )

        # Print summary using utility
        log_progress("\n=== KG Update Summary ===")
        log_progress(f"Total documents: {batch_result.total_documents}")
        log_progress(f"Successful: {batch_result.successful_documents}")
        log_progress(f"Failed: {batch_result.failed_documents}")
        log_progress(
            f"Average time per doc: {batch_result.avg_time_per_document_ms:.2f}ms"
        )

        # Write results to output file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(batch_result.model_dump(), f, indent=2)
            log_progress(f"Results saved to: {output_path}")

        # Write batch result to stdout as JSON
        output_json(batch_result)

    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
