#!/usr/bin/env python3
"""Knowledge Graph Update Script.

Updates the knowledge graph by processing documents and extracting
entities and relations.

Usage:
    python run_kg_update.py --domain movie --progress-path results/progress.json
    python run_kg_update.py --dataset data/corpus.jsonl.bz2 --num-workers 64
    python run_kg_update.py --mock --verbose
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mellea_contribs.kg.kgrag import orchestrate_kg_update
from mellea_contribs.kg.updater_models import (
    KGUpdateRunConfig,
    UpdateBatchResult,
    UpdateResult,
    UpdateStats,
)
from mellea_contribs.kg.utils import (
    BaseProgressLogger,
    create_backend,
    create_session_from_env,
    load_jsonl,
    log_progress,
    output_json,
    print_stats,
    setup_logging,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Update knowledge graph from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dataset data/corpus.jsonl.bz2 --mock
  %(prog)s --num-workers 32 --queue-size 32
  %(prog)s --domain movie --progress-path results/progress.json
  %(prog)s --verbose --mock
        """,
    )

    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to dataset file (overrides env KG_BASE_DIRECTORY)",
    )
    parser.add_argument(
        "--domain", type=str, default="movie",
        help="Knowledge domain (default: movie)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=64,
        help="Number of concurrent workers (default: 64)",
    )
    parser.add_argument(
        "--queue-size", type=int, default=64,
        help="Queue size for data loading (default: 64)",
    )
    parser.add_argument(
        "--db-uri", type=str,
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Graph database connection URI (default: $NEO4J_URI or bolt://localhost:7687)",
    )
    parser.add_argument(
        "--db-user", type=str,
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Graph database username (default: $NEO4J_USER or neo4j)",
    )
    parser.add_argument(
        "--db-password", type=str,
        default=os.getenv("NEO4J_PASSWORD", "password"),
        help="Graph database password (default: $NEO4J_PASSWORD or password)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use MockGraphBackend instead of the graph database",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--extraction-loop-budget", type=int, default=3,
        help="Entity/relation extraction loop budget (default: 3)",
    )
    parser.add_argument(
        "--alignment-loop-budget", type=int, default=2,
        help="Alignment refinement loop budget (default: 2)",
    )
    parser.add_argument(
        "--align-topk", type=int, default=10,
        help="Number of top candidates for alignment (default: 10)",
    )
    parser.add_argument(
        "--progress-path", type=str, default="results/update_kg_progress.json",
        help="Progress log file path",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


async def process_document(
    doc_id: str,
    text: str,
    backend: Any,
    session: Any,
    domain: str,
    model: str,
    progress_tracker: BaseProgressLogger,
) -> UpdateResult:
    """Process a single document and update the KG.

    Args:
        doc_id: Document ID.
        text: Document text.
        backend: Graph backend.
        session: Mellea session.
        domain: Knowledge domain.
        model: LLM model name.
        progress_tracker: Progress tracker.

    Returns:
        UpdateResult with processing details.
    """
    start_time = time.perf_counter()
    log_progress(f"[{doc_id[:12]}] Starting...")

    try:
        update_result = await orchestrate_kg_update(
            session=session,
            backend=backend,
            doc_text=text,
            domain=domain,
            hints="",
            entity_types="",
            relation_types="",
        )

        elapsed_time = time.perf_counter() - start_time
        entities_found = len(update_result.get("extracted_entities", []))
        relations_found = len(update_result.get("extracted_relations", []))

        result = UpdateResult(
            document_id=doc_id,
            success=True,
            entities_found=entities_found,
            relations_found=relations_found,
            entities_added=entities_found,
            relations_added=relations_found,
            processing_time_ms=elapsed_time * 1000,
            model_used=model,
        )

        log_progress(
            f"[{doc_id[:12]}] Done — {entities_found} entities, "
            f"{relations_found} relations ({elapsed_time * 1000:.0f}ms)"
        )

        progress_tracker.add_stat({
            "doc_id": doc_id,
            "entities_extracted": entities_found,
            "entities_new": entities_found,
            "relations_extracted": relations_found,
            "relations_new": relations_found,
            "processing_time": round(elapsed_time, 2),
        })
        progress_tracker.mark_processed(doc_id)

        return result

    except Exception as e:
        import traceback
        elapsed_time = time.perf_counter() - start_time
        log_progress(f"[{doc_id[:12]}] ERROR: {type(e).__name__}: {e}", level="ERROR")
        log_progress(traceback.format_exc(), level="DEBUG")
        return UpdateResult(
            document_id=doc_id,
            success=False,
            error=str(e),
            processing_time_ms=elapsed_time * 1000,
            model_used=model,
        )


async def process_dataset(
    dataset_path: Path,
    config: KGUpdateRunConfig,
    progress_tracker: BaseProgressLogger,
) -> UpdateBatchResult:
    """Process the entire dataset with parallel workers.

    Args:
        dataset_path: Path to the JSONL dataset file.
        config: Run configuration.
        progress_tracker: Progress tracker.

    Returns:
        Aggregated batch result.
    """
    backend = create_backend(
        backend_type="neo4j" if not config.mock else "mock",
        neo4j_uri=config.db_uri,
        neo4j_user=config.db_user,
        neo4j_password=config.db_password,
    )

    session, model_id = create_session_from_env(default_model=config.model)
    log_progress(f"Using model: {model_id}, API base: {os.getenv('API_BASE') or '(default)'}")

    batch_result = UpdateBatchResult()
    tasks = []
    semaphore = asyncio.Semaphore(config.num_workers)

    async def process_with_semaphore(doc_id: str, text: str) -> UpdateResult:
        async with semaphore:
            return await process_document(
                doc_id=doc_id, text=text, backend=backend, session=session,
                domain=config.domain, model=model_id, progress_tracker=progress_tracker,
            )

    try:
        doc_num = 0
        for doc_num, doc in enumerate(load_jsonl(dataset_path), 1):
            doc_id = doc.get("id") or doc.get("interaction_id") or f"doc_{doc_num}"
            text = doc.get("text") or doc.get("query") or doc.get("context") or ""
            if not text:
                log_progress(f"[{doc_num}] WARNING: Empty text for {doc_id}")
                continue
            tasks.append(process_with_semaphore(doc_id, text))

        total_tasks = len(tasks)
        completed_count = 0

        async def _tracked(coro: Any) -> UpdateResult:
            nonlocal completed_count
            result = await coro
            completed_count += 1
            status = "✓" if result.success else "✗"
            log_progress(f"[{completed_count}/{total_tasks}] {status} {result.document_id[:12]}")
            return result

        log_progress(f"Processing {total_tasks} documents with {config.num_workers} workers...")
        results = list(await asyncio.gather(*[_tracked(t) for t in tasks]))

        for result in results:
            if result.success:
                batch_result.successful_documents += 1
            else:
                batch_result.failed_documents += 1

    finally:
        await backend.close()

    batch_result.total_documents = len(results)
    batch_result.results = results

    if results:
        stats = UpdateStats()
        stats.total_documents = len(results)
        stats.successful_documents = batch_result.successful_documents
        stats.failed_documents = batch_result.failed_documents
        for result in results:
            stats.entities_extracted += result.entities_found
            stats.relations_extracted += result.relations_found
            stats.entities_new += result.entities_added
            stats.relations_new += result.relations_added
        batch_result.total_time_ms = sum(r.processing_time_ms for r in results)
        batch_result.avg_time_per_document_ms = (
            batch_result.total_time_ms / len(results)
        )
        stats.total_processing_time_ms = batch_result.total_time_ms
        stats.average_processing_time_per_doc_ms = batch_result.avg_time_per_document_ms
        batch_result.stats = stats

    return batch_result


def load_env_file() -> None:
    """Load environment variables from .env file in parent directory."""
    script_dir = Path(__file__).parent
    env_path = script_dir.parent / ".env"
    if env_path.exists():
        log_progress(f"Loading environment from: {env_path}")
        load_dotenv(env_path, override=False)
    else:
        log_progress(f"⚠️  .env not found at {env_path} (optional)")


async def main() -> int:
    """Main async entry point."""
    setup_logging(log_level="INFO")
    load_env_file()
    args = parse_arguments()

    config = KGUpdateRunConfig(
        model=args.model,
        num_workers=args.num_workers,
        queue_size=args.queue_size,
        extraction_loop_budget=args.extraction_loop_budget,
        alignment_loop_budget=args.alignment_loop_budget,
        align_topk=args.align_topk,
        domain=args.domain,
        progress_path=args.progress_path,
        db_uri=args.db_uri,
        db_user=args.db_user,
        db_password=args.db_password,
        mock=args.mock,
        verbose=args.verbose,
    )

    # Resolve dataset path
    if args.dataset:
        config.dataset_path = args.dataset
    else:
        base_dir = os.getenv(
            "KG_BASE_DIRECTORY",
            os.path.join(os.path.dirname(__file__), "..", "dataset"),
        )
        config.dataset_path = os.path.join(base_dir, "crag_movie_dev.jsonl.bz2")

    if not Path(config.dataset_path).exists():
        log_progress(f"ERROR: Dataset not found: {config.dataset_path}")
        return 1

    try:
        progress_tracker = BaseProgressLogger(config.progress_path)
        progress_tracker.load()
        if progress_tracker.num_processed:
            log_progress(f"Resuming: {progress_tracker.num_processed} documents already processed.")

        log_progress("=" * 60)
        log_progress("KG Update Configuration:")
        log_progress("=" * 60)
        log_progress(f"Dataset: {config.dataset_path}")
        log_progress(f"Domain: {config.domain}")
        log_progress(f"Workers: {config.num_workers}")
        log_progress(f"Queue size: {config.queue_size}")
        log_progress(f"Extraction loop budget: {config.extraction_loop_budget}")
        log_progress(f"Alignment loop budget: {config.alignment_loop_budget}")
        log_progress(f"Top-K candidates: {config.align_topk}")
        log_progress(f"Model: {config.model}")
        log_progress(f"Backend: {'Mock' if config.mock else 'Graph DB'}")
        log_progress(f"Progress: {config.progress_path}")
        log_progress("=" * 60)

        Path("results").mkdir(exist_ok=True)

        log_progress("Starting KG update...")
        batch_result = await process_dataset(Path(config.dataset_path), config, progress_tracker)

        progress_tracker.save()

        log_progress("=" * 60)
        log_progress("✅ KG Update Completed Successfully!")
        log_progress("=" * 60)
        log_progress(f"Processed documents: {batch_result.total_documents}")
        log_progress(f"Successful: {batch_result.successful_documents}")
        log_progress(f"Failed: {batch_result.failed_documents}")
        if batch_result.stats:
            log_progress(f"Total entities: {batch_result.stats.entities_extracted}")
            log_progress(f"Total relations: {batch_result.stats.relations_extracted}")
        log_progress(f"Average time per doc: {batch_result.avg_time_per_document_ms:.2f}ms")
        log_progress(f"Progress saved to: {config.progress_path}")
        log_progress("=" * 60)

        if batch_result.stats:
            print_stats(batch_result.stats)
        output_json(batch_result)

        return 0

    except KeyboardInterrupt:
        log_progress("\n⚠️  KG update interrupted by user")
        return 130
    except Exception as e:
        log_progress(f"❌ KG update failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
