#!/usr/bin/env python3
"""Knowledge Graph Update Script

This script updates the knowledge graph by processing documents and extracting
entities and relations using modern patterns.

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
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from mellea_contribs.kg.kgrag import orchestrate_kg_update
from mellea_contribs.kg.updater_models import (
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


class SessionConfig:
    """Configuration for LLM session settings."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
    ):
        """Initialize session configuration."""
        self.model = model


class UpdaterConfig:
    """Configuration for KG updater settings."""

    def __init__(
        self,
        num_workers: int = 64,
        queue_size: int = 64,
        extraction_loop_budget: int = 3,
        alignment_loop_budget: int = 2,
        align_topk: int = 10,
        align_entity: bool = True,
        merge_entity: bool = True,
        align_relation: bool = True,
        merge_relation: bool = True,
    ):
        """Initialize updater configuration."""
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.extraction_loop_budget = extraction_loop_budget
        self.alignment_loop_budget = alignment_loop_budget
        self.align_topk = align_topk
        self.align_entity = align_entity
        self.merge_entity = merge_entity
        self.align_relation = align_relation
        self.merge_relation = merge_relation


class DatasetConfig:
    """Configuration for dataset settings."""

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        domain: str = "movie",
        progress_path: str = "results/update_kg_progress.json",
    ):
        """Initialize dataset configuration."""
        self.dataset_path = dataset_path
        self.domain = domain
        self.progress_path = progress_path

    def validate(self) -> bool:
        """Validate configuration."""
        if not self.dataset_path:
            base_dir = os.getenv(
                "KG_BASE_DIRECTORY",
                os.path.join(os.path.dirname(__file__), "..", "dataset"),
            )
            self.dataset_path = os.path.join(base_dir, "crag_movie_dev.jsonl.bz2")

        if not Path(self.dataset_path).exists():
            log_progress(f"ERROR: Dataset not found: {self.dataset_path}")
            return False

        return True


class KGUpdateConfig:
    """Unified configuration for KG update operations."""

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        domain: str = "movie",
        num_workers: int = 64,
        queue_size: int = 64,
        progress_path: str = "results/update_kg_progress.json",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        mock: bool = False,
        model: str = "gpt-4o-mini",
        extraction_loop_budget: int = 3,
        alignment_loop_budget: int = 2,
        align_topk: int = 10,
        verbose: bool = False,
    ):
        """Initialize configuration."""
        self.session_config = SessionConfig(model=model)
        self.updater_config = UpdaterConfig(
            num_workers=num_workers,
            queue_size=queue_size,
            extraction_loop_budget=extraction_loop_budget,
            alignment_loop_budget=alignment_loop_budget,
            align_topk=align_topk,
        )
        self.dataset_config = DatasetConfig(
            dataset_path=dataset_path,
            domain=domain,
            progress_path=progress_path,
        )
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.mock = mock
        self.verbose = verbose

    def validate(self) -> bool:
        """Validate configuration."""
        return self.dataset_config.validate()




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

    # Dataset configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset file (overrides env KG_BASE_DIRECTORY)",
    )
    
    parser.add_argument(
        "--domain",
        type=str,
        default="movie",
        help="Knowledge domain (default: movie)",
    )

    # Worker configuration
    parser.add_argument(
        "--num-workers",
        type=int,
        default=64,
        help="Number of concurrent workers (default: 64)",
    )

    parser.add_argument(
        "--queue-size",
        type=int,
        default=64,
        help="Queue size for data loading (default: 64)",
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
        help="Use MockGraphBackend instead of Neo4j",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )

    # Alignment and merge configuration
    parser.add_argument(
        "--extraction-loop-budget",
        type=int,
        default=3,
        help="Entity/relation extraction loop budget (default: 3)",
    )

    parser.add_argument(
        "--alignment-loop-budget",
        type=int,
        default=2,
        help="Alignment refinement loop budget (default: 2)",
    )

    parser.add_argument(
        "--align-topk",
        type=int,
        default=10,
        help="Number of top candidates for alignment (default: 10)",
    )

    # Progress tracking
    parser.add_argument(
        "--progress-path",
        type=str,
        default="results/update_kg_progress.json",
        help="Progress log file path",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


# Worker-local storage for session and backend instances
_worker_instances: Dict[str, tuple] = {}


async def process_document(
    doc_id: str,
    text: str,
    backend: Any,
    session: Any,
    domain: str,
    model: str,
    progress_tracker: BaseProgressLogger,
) -> UpdateResult:
    """Process a single document.

    Args:
        doc_id: Document ID
        text: Document text
        backend: Graph backend
        session: Mellea session
        domain: Knowledge domain
        model: LLM model name
        progress_tracker: Progress tracker

    Returns:
        UpdateResult with processing details
    """
    start_time = time.perf_counter()
    log_progress(f"[{doc_id[:12]}] Starting...")

    try:
        # Call orchestrate_kg_update
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

        # Extract statistics
        entities_found = len(update_result.get("extracted_entities", []))
        relations_found = len(update_result.get("extracted_relations", []))

        # Create result
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

        # Track progress
        progress_tracker.add_stat(
            {
                "doc_id": doc_id,
                "entities_extracted": entities_found,
                "entities_new": entities_found,
                "relations_extracted": relations_found,
                "relations_new": relations_found,
                "processing_time": round(elapsed_time, 2),
            }
        )
        progress_tracker.mark_processed(doc_id)

        return result

    except Exception as e:
        import traceback
        elapsed_time = time.perf_counter() - start_time
        log_progress(f"[{doc_id[:12]}] ERROR: {type(e).__name__}: {e}", level="ERROR")
        log_progress(traceback.format_exc(), level="DEBUG")

        result = UpdateResult(
            document_id=doc_id,
            success=False,
            error=str(e),
            processing_time_ms=elapsed_time * 1000,
            model_used=model,
        )

        return result


async def process_dataset(
    dataset_path: Path,
    config: KGUpdateConfig,
    progress_tracker: BaseProgressLogger,
) -> UpdateBatchResult:
    """Process entire dataset with parallel workers.

    Args:
        dataset_path: Path to dataset file
        config: Update configuration
        progress_tracker: Progress tracker

    Returns:
        Batch result with all document results
    """
    # Create shared backend and session
    backend = create_backend(
        backend_type="neo4j" if not config.mock else "mock",
        neo4j_uri=config.neo4j_uri,
        neo4j_user=config.neo4j_user,
        neo4j_password=config.neo4j_password,
    )

    # Create session with API configuration from environment
    session, model_id = create_session_from_env(default_model=config.session_config.model)
    log_progress(f"Using model: {model_id}, API base: {os.getenv('API_BASE') or '(default)'}")

    batch_result = UpdateBatchResult()
    results = []
    tasks = []

    # Semaphore to limit concurrent workers
    semaphore = asyncio.Semaphore(config.updater_config.num_workers)

    async def process_with_semaphore(doc_id: str, text: str) -> UpdateResult:
        """Process document with semaphore for concurrency control."""
        async with semaphore:
            return await process_document(
                doc_id=doc_id,
                text=text,
                backend=backend,
                session=session,
                domain=config.dataset_config.domain,
                model=model_id,
                progress_tracker=progress_tracker,
            )

    try:
        # Create tasks for all documents
        doc_num = 0
        for doc_num, doc in enumerate(load_jsonl(dataset_path), 1):
            # Handle different dataset formats
            doc_id = doc.get("id") or doc.get("interaction_id") or f"doc_{doc_num}"
            # Try different text field names
            text = doc.get("text") or doc.get("query") or doc.get("context") or ""

            if not text:
                log_progress(f"[{doc_num}] WARNING: Empty text for {doc_id}")
                continue

            task = process_with_semaphore(doc_id, text)
            tasks.append(task)

        total_tasks = len(tasks)
        completed_count = 0

        async def _tracked(coro: Any) -> UpdateResult:
            nonlocal completed_count
            result = await coro
            completed_count += 1
            status = "✓" if result.success else "✗"
            log_progress(f"[{completed_count}/{total_tasks}] {status} {result.document_id[:12]}")
            return result

        log_progress(f"Processing {total_tasks} documents with {config.updater_config.num_workers} workers...")
        results = list(await asyncio.gather(*[_tracked(t) for t in tasks]))

        # Aggregate results
        for result in results:
            if result.success:
                batch_result.successful_documents += 1
            else:
                batch_result.failed_documents += 1

    finally:
        await backend.close()

    # Compute batch statistics
    batch_result.total_documents = len(results)
    batch_result.results = results

    if len(results) > 0:
        # Create aggregated stats
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
        # Mirror timing into UpdateStats so print_stats shows real values
        stats.total_processing_time_ms = batch_result.total_time_ms
        stats.average_processing_time_per_doc_ms = batch_result.avg_time_per_document_ms
        batch_result.stats = stats

    return batch_result


def load_env_file() -> None:
    """Load environment variables from .env file in parent directory."""
    # Try to load .env from parent directory (kgrag root)
    script_dir = Path(__file__).parent
    env_path = script_dir.parent / ".env"

    if env_path.exists():
        log_progress(f"Loading environment from: {env_path}")
        load_dotenv(env_path, override=False)
    else:
        log_progress(f"⚠️  .env not found at {env_path} (optional)")




async def main() -> int:
    """Main async entry point."""
    # Initialise logging early so all log_progress calls are visible
    setup_logging(log_level="INFO")

    # Load environment variables from .env file
    load_env_file()

    args = parse_arguments()

    # Create configuration
    config = KGUpdateConfig(
        dataset_path=args.dataset,
        domain=args.domain,
        num_workers=args.num_workers,
        queue_size=args.queue_size,
        progress_path=args.progress_path,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        mock=args.mock,
        model=args.model,
        extraction_loop_budget=args.extraction_loop_budget,
        alignment_loop_budget=args.alignment_loop_budget,
        align_topk=args.align_topk,
        verbose=args.verbose,
    )

    try:
        # Validate configuration
        if not config.validate():
            return 1

        # Create progress tracker
        progress_tracker = BaseProgressLogger(config.dataset_config.progress_path)
        progress_tracker.load()
        if progress_tracker.num_processed:
            log_progress(f"Resuming: {progress_tracker.num_processed} documents already processed.")

        # Log configuration
        log_progress("=" * 60)
        log_progress("KG Update Configuration:")
        log_progress("=" * 60)
        log_progress(f"Dataset: {config.dataset_config.dataset_path}")
        log_progress(f"Domain: {config.dataset_config.domain}")
        log_progress(f"Workers: {config.updater_config.num_workers}")
        log_progress(f"Queue size: {config.updater_config.queue_size}")
        log_progress(f"Extraction loop budget: {config.updater_config.extraction_loop_budget}")
        log_progress(f"Alignment loop budget: {config.updater_config.alignment_loop_budget}")
        log_progress(f"Top-K candidates: {config.updater_config.align_topk}")
        log_progress(f"Model: {config.session_config.model}")
        log_progress(f"Backend: {'Mock' if config.mock else 'Neo4j'}")
        log_progress(f"Progress: {config.dataset_config.progress_path}")
        log_progress("=" * 60)

        # Ensure results directory exists
        Path("results").mkdir(exist_ok=True)

        # Process dataset
        log_progress("Starting KG update...")
        dataset_path = Path(config.dataset_config.dataset_path)
        batch_result = await process_dataset(dataset_path, config, progress_tracker)

        # Save progress
        progress_tracker.save()

        # Log results
        log_progress("=" * 60)
        log_progress("✅ KG Update Completed Successfully!")
        log_progress("=" * 60)
        log_progress(f"Processed documents: {batch_result.total_documents}")
        log_progress(f"Successful: {batch_result.successful_documents}")
        log_progress(f"Failed: {batch_result.failed_documents}")
        if batch_result.stats:
            log_progress(f"Total entities: {batch_result.stats.entities_extracted}")
            log_progress(f"Total relations: {batch_result.stats.relations_extracted}")
        log_progress(
            f"Average time per doc: {batch_result.avg_time_per_document_ms:.2f}ms"
        )
        log_progress(f"Progress saved to: {config.dataset_config.progress_path}")
        log_progress("=" * 60)

        # Print stats and output JSON
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
