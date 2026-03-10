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
import bz2
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

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


class KGUpdateConfig:
    """Configuration for KG update operations."""

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
        verbose: bool = False,
    ):
        """Initialize configuration."""
        self.dataset_path = dataset_path
        self.domain = domain
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.progress_path = progress_path
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.mock = mock
        self.model = model
        self.verbose = verbose

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


class KGProgressTracker:
    """Track progress of KG updates."""

    def __init__(self, progress_path: str = "results/update_kg_progress.json"):
        """Initialize progress tracker."""
        self.progress_path = progress_path
        self.stats: list[Dict[str, Any]] = []
        self.processed_docs: set[str] = set()
        self.start_time = time.time()
        self.load_progress()

    def load_progress(self) -> None:
        """Load existing progress from file."""
        if Path(self.progress_path).exists():
            try:
                with open(self.progress_path, "r") as f:
                    data = json.load(f)
                    self.stats = data.get("stats", [])
                    self.processed_docs = set(data.get("processed_docs", []))
                    log_progress(f"Loaded progress: {len(self.processed_docs)} documents")
            except Exception as e:
                log_progress(f"Warning: Could not load progress: {e}")

    def add_stat(self, stat: Dict[str, Any]) -> None:
        """Add a statistic."""
        self.stats.append(stat)
        self.processed_docs.add(stat.get("doc_id", ""))

    def save_progress(self) -> None:
        """Save progress to file."""
        Path(self.progress_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_path, "w") as f:
            json.dump(
                {
                    "stats": self.stats,
                    "processed_docs": list(self.processed_docs),
                    "last_update": datetime.now().isoformat(),
                    "elapsed_time": time.time() - self.start_time,
                },
                f,
                indent=2,
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_entities = sum(s.get("entities_new", 0) for s in self.stats)
        total_relations = sum(s.get("relations_new", 0) for s in self.stats)
        total_time = sum(s.get("processing_time", 0) for s in self.stats)

        return {
            "processed_documents": len(self.stats),
            "total_entities": total_entities,
            "total_relations": total_relations,
            "total_processing_time": round(total_time, 2),
            "elapsed_time": round(time.time() - self.start_time, 2),
        }


def load_jsonl_compressed(file_path: Path):
    """Load JSONL file that may be compressed (.bz2).

    Args:
        file_path: Path to JSONL or JSONL.bz2 file

    Yields:
        Parsed JSON objects
    """
    if str(file_path).endswith('.bz2'):
        with bz2.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        for obj in load_jsonl(file_path):
            yield obj


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
    progress_tracker: KGProgressTracker,
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

        return result

    except Exception as e:
        elapsed_time = time.perf_counter() - start_time
        log_progress(f"[{doc_id}] ERROR: {e}")

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
    progress_tracker: KGProgressTracker,
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
    session = create_session(model_id=config.model)

    batch_result = UpdateBatchResult()
    results = []
    tasks = []

    # Semaphore to limit concurrent workers
    semaphore = asyncio.Semaphore(config.num_workers)

    async def process_with_semaphore(doc_id: str, text: str) -> UpdateResult:
        """Process document with semaphore for concurrency control."""
        async with semaphore:
            return await process_document(
                doc_id=doc_id,
                text=text,
                backend=backend,
                session=session,
                domain=config.domain,
                model=config.model,
                progress_tracker=progress_tracker,
            )

    try:
        # Create tasks for all documents
        doc_num = 0
        for doc_num, doc in enumerate(load_jsonl_compressed(dataset_path), 1):
            # Handle different dataset formats
            doc_id = doc.get("id") or doc.get("interaction_id") or f"doc_{doc_num}"
            # Try different text field names
            text = doc.get("text") or doc.get("query") or doc.get("context") or ""

            if not text:
                log_progress(f"[{doc_num}] WARNING: Empty text for {doc_id}")
                continue

            task = process_with_semaphore(doc_id, text)
            tasks.append(task)

        # Process all tasks concurrently
        log_progress(f"Processing {len(tasks)} documents with {config.num_workers} workers...")
        results = await asyncio.gather(*tasks)

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

        batch_result.stats = stats
        batch_result.total_time_ms = sum(r.processing_time_ms for r in results)
        batch_result.avg_time_per_document_ms = (
            batch_result.total_time_ms / len(results)
        )

    return batch_result


async def main() -> int:
    """Main async entry point."""
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
        verbose=args.verbose,
    )

    try:
        # Validate configuration
        if not config.validate():
            return 1

        # Create progress tracker
        progress_tracker = KGProgressTracker(config.progress_path)

        # Log configuration
        log_progress("=" * 60)
        log_progress("KG Update Configuration:")
        log_progress("=" * 60)
        log_progress(f"Dataset: {config.dataset_path}")
        log_progress(f"Domain: {config.domain}")
        log_progress(f"Workers: {config.num_workers}")
        log_progress(f"Queue size: {config.queue_size}")
        log_progress(f"Model: {config.model}")
        log_progress(f"Backend: {'Mock' if config.mock else 'Neo4j'}")
        log_progress(f"Progress: {config.progress_path}")
        log_progress("=" * 60)

        # Ensure results directory exists
        Path("results").mkdir(exist_ok=True)

        # Process dataset
        log_progress("Starting KG update...")
        dataset_path = Path(config.dataset_path)
        batch_result = await process_dataset(dataset_path, config, progress_tracker)

        # Save progress
        progress_tracker.save_progress()

        # Get summary
        summary = progress_tracker.get_summary()

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
        log_progress(f"Progress saved to: {config.progress_path}")
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
