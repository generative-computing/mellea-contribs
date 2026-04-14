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
from typing import Any, Dict

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from mellea_contribs.kg.kgrag import orchestrate_kg_update
from mellea_contribs.kg.updater_models import (
    KGUpdateRunConfig,
    UpdateBatchResult,
    UpdateResult,
    UpdateStats,
)
from mellea_contribs.kg.utils import (
    BaseProgressLogger,
    add_graph_args,
    aggregate_update_results,
    create_backend,
    create_session_from_env,
    log_progress,
    output_json,
    print_stats,
    setup_logging,
)

from dataset.update_dataset_loader import UpdateDatasetLoader

try:
    from preprocessor.movie_preprocessor import MovieKGPreprocessor
except ImportError:
    MovieKGPreprocessor = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Per-document processing
# ---------------------------------------------------------------------------


async def process_document(
    item: Dict[str, Any],
    *,
    preprocessor: Any,
    backend: Any,
    model_id: str,
    progress_tracker: BaseProgressLogger,
) -> UpdateResult:
    """Process a single document and update the KG.

    Args:
        item: Normalised item from ``UpdateDatasetLoader``.
        preprocessor: Domain-specific KGPreprocessor.
        backend: Graph backend.
        model_id: LLM model name (recorded in result).
        progress_tracker: Progress tracker.

    Returns:
        UpdateResult with processing details.
    """
    doc_id = item["id"]
    text = item["text"]
    start_time = time.perf_counter()
    log_progress(f"[{doc_id[:12]}] Starting...")

    try:
        update_result = await orchestrate_kg_update(
            preprocessor=preprocessor,
            backend=backend,
            doc_text=text,
        )

        elapsed_time = time.perf_counter() - start_time
        entities_found = len(update_result.get("extracted_entities", []))
        relations_found = len(update_result.get("extracted_relations", []))

        result = UpdateResult(
            document_id=doc_id,
            success=True,
            entities_found=entities_found,
            relations_found=relations_found,
            entities_added=update_result.get("entities_inserted", 0),
            relations_added=update_result.get("relations_inserted", 0),
            processing_time_ms=elapsed_time * 1000,
            model_used=model_id,
        )

        log_progress(
            f"[{doc_id[:12]}] Done — {entities_found} entities, "
            f"{relations_found} relations ({elapsed_time * 1000:.0f}ms)"
        )
        progress_tracker.add_stat({
            "doc_id": doc_id,
            "entities_extracted": entities_found,
            "relations_extracted": relations_found,
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
            model_used=model_id,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
    add_graph_args(parser)
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


async def main() -> int:
    """Main async entry point."""
    setup_logging(log_level="INFO")

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

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
        graph_uri=args.db_uri,
        graph_user=args.db_user,
        graph_password=args.db_password,
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
            log_progress(
                f"Resuming: {progress_tracker.num_processed} documents already processed."
            )

        log_progress("=" * 60)
        log_progress("KG Update Configuration:")
        log_progress(f"  Dataset:                {config.dataset_path}")
        log_progress(f"  Domain:                 {config.domain}")
        log_progress(f"  Workers:                {config.num_workers}")
        log_progress(f"  Queue size:             {config.queue_size}")
        log_progress(f"  Extraction loop budget: {config.extraction_loop_budget}")
        log_progress(f"  Alignment loop budget:  {config.alignment_loop_budget}")
        log_progress(f"  Top-K candidates:       {config.align_topk}")
        log_progress(f"  Model:                  {config.model}")
        log_progress(f"  Backend:                {'Mock' if config.mock else 'Graph DB'}")
        log_progress(f"  Progress:               {config.progress_path}")
        log_progress("=" * 60)

        Path("results").mkdir(exist_ok=True)

        backend = create_backend(
            backend_type="mock" if config.mock else "neo4j",
            neo4j_uri=config.graph_uri,
            neo4j_user=config.graph_user,
            neo4j_password=config.graph_password,
        )
        session, model_id = create_session_from_env(default_model=config.model)
        log_progress(
            f"Using model: {model_id}, "
            f"API base: {os.getenv('API_BASE') or '(default)'}"
        )

        if MovieKGPreprocessor is None:
            log_progress("ERROR: MovieKGPreprocessor not available", level="ERROR")
            return 1
        preprocessor = MovieKGPreprocessor(backend=backend, session=session)

        async def _process(item: Dict[str, Any]) -> UpdateResult:
            return await process_document(
                item,
                preprocessor=preprocessor,
                backend=backend,
                model_id=model_id,
                progress_tracker=progress_tracker,
            )

        log_progress("Starting KG update...")
        results: list[UpdateResult] = []
        try:
            loader = UpdateDatasetLoader(
                dataset_path=config.dataset_path,
                num_workers=config.num_workers,
                queue_size=config.queue_size,
            )
            results = await loader.run(
                process_fn=_process,
                id_key="id",
                skip_ids=progress_tracker.processed_ids,
            )
        finally:
            await backend.close()

        progress_tracker.save()

        # Aggregate stats
        stats: UpdateStats = aggregate_update_results(results)

        batch_result = UpdateBatchResult(
            total_documents=stats.total_documents,
            successful_documents=stats.successful_documents,
            failed_documents=stats.failed_documents,
            total_time_ms=stats.total_processing_time_ms,
            avg_time_per_document_ms=stats.average_processing_time_per_doc_ms,
            results=results,
            stats=stats,
        )

        log_progress("=" * 60)
        log_progress("KG Update Completed!")
        log_progress(f"  Processed:       {batch_result.total_documents}")
        log_progress(f"  Successful:      {batch_result.successful_documents}")
        log_progress(f"  Failed:          {batch_result.failed_documents}")
        log_progress(f"  Total entities:  {stats.entities_extracted}")
        log_progress(f"  Total relations: {stats.relations_extracted}")
        log_progress(
            f"  Avg time/doc:    {batch_result.avg_time_per_document_ms:.2f}ms"
        )
        log_progress(f"  Progress saved:  {config.progress_path}")
        log_progress("=" * 60)

        print_stats(stats)
        output_json(batch_result)
        return 0

    except KeyboardInterrupt:
        log_progress("\nKG update interrupted by user")
        return 130
    except Exception as e:
        log_progress(f"KG update failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
