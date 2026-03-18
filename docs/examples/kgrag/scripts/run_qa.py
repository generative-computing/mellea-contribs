#!/usr/bin/env python3
"""Run QA on CRAG movie questions via the Think-on-Graph pipeline.

Reads questions from a JSONL file, answers each one using
``orchestrate_qa_retrieval``, and writes per-question results to an output
JSONL file.  Progress is persisted so interrupted runs can resume from where
they left off.

Three independent sessions are created:

* **main session** — question decomposition, entity alignment, relation /
  triplet pruning.
* **eval session** — knowledge sufficiency evaluation, consensus validation,
  direct-answer fallback.  Defaults to the main session when not configured.
* **embedding client** — async OpenAI-compatible client for vector-based
  entity alignment.  Optional; falls back to fuzzy name search only.

Configuration is driven by environment variables so the script works
transparently in any containerised environment.  The variable names
mirror those used by ``run_kg_update.py``:

.. code-block:: bash

    # Required — any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama, Azure, etc.)
    export API_BASE=https://your-llm-endpoint/v1
    export API_KEY=your-api-key
    export MODEL_NAME=meta-llama/llama-3-70b-instruct   # or gpt-4o-mini etc.

    # Optional — separate eval model (defaults to main session)
    export EVAL_API_BASE=...
    export EVAL_API_KEY=...
    export EVAL_MODEL_NAME=...

    # Optional — embedding model for vector entity alignment
    export EMB_API_BASE=...
    export EMB_API_KEY=...
    export EMB_MODEL_NAME=text-embedding-3-small

When ``API_BASE`` is set the session uses ``OpenAIBackend`` directly, which
works for any OpenAI-compatible endpoint (OpenAI, vLLM, Azure, Ollama, etc.)
regardless of model ID.  Use ``--mock`` to skip LLM calls entirely during
local testing.

    python run_qa.py \\
        --dataset ../dataset/crag_movie_tiny.jsonl.bz2 \\
        --output /tmp/qa_results.jsonl \\
        --progress /tmp/qa_progress.json \\
        --mock

Output JSONL format (one JSON object per line)::

    {
        "id": "q_0",
        "query": "Who directed Inception?",
        "query_time": "2024-03-05",
        "predicted": "Christopher Nolan",
        "answer": "Christopher Nolan",
        "answer_aliases": ["Christopher Nolan", "Nolan"],
        "correct": true,
        "eval_method": "exact",
        "elapsed_ms": 1234.5
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from mellea_contribs.kg.kgrag import orchestrate_qa_retrieval
from mellea_contribs.kg.utils import (
    QAProgressLogger,
    create_backend,
    create_embedding_client,
    create_session_from_env,
    evaluate_predictions,
    log_progress,
    setup_logging,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset.movie_dataset_loader import MovieDatasetLoader

# ---------------------------------------------------------------------------
# Session / config helpers
# ---------------------------------------------------------------------------

_HINTS = (
    "This is a movie-domain knowledge graph containing Movies, Persons, Awards, "
    "and Year nodes.  Common relation types: acted_in, directed_by, produced_by, "
    "nominated_for, won, released_in."
)

_DEFAULT_MODEL = "gpt-4o-mini"




def create_emb_client_optional():
    """Create an embedding client if ``EMB_API_BASE`` is set."""
    api_base = os.getenv("EMB_API_BASE")
    if not api_base:
        return None
    return create_embedding_client(
        api_base=api_base,
        api_key=os.getenv("EMB_API_KEY", "dummy"),
        model_name=os.getenv("EMB_MODEL_NAME", "text-embedding-3-small"),
    )


# ---------------------------------------------------------------------------
# Per-question processing
# ---------------------------------------------------------------------------


async def process_question(
    item: Dict[str, Any],
    *,
    backend,
    session,
    eval_session,
    emb_client,
    domain: str,
    num_routes: int,
    width: int,
    depth: int,
) -> Dict[str, Any]:
    """Answer one QA item and return a result dict.

    Args:
        item: Normalised QA item from ``MovieDatasetLoader``.
        backend: Graph database backend.
        session: Primary Mellea session.
        eval_session: Eval Mellea session (may equal ``session``).
        emb_client: Optional embedding client.
        domain: Domain hint string.
        num_routes: Number of solving routes.
        width: ToG traversal width.
        depth: ToG traversal depth.

    Returns:
        Result dict with ``id``, ``query``, ``predicted``, timing, and
        correctness fields.
    """
    t0 = time.perf_counter()
    error: Optional[str] = None
    predicted = ""

    try:
        predicted = await orchestrate_qa_retrieval(
            session=session,
            backend=backend,
            query=item["query"],
            query_time=item.get("query_time", ""),
            domain=domain,
            num_routes=num_routes,
            hints=_HINTS,
            eval_session=eval_session,
            emb_client=emb_client,
            width=width,
            depth=depth,
        )
    except Exception as exc:
        error = str(exc)
        log_progress(f"  ERROR [{item['id']}]: {exc}", level="WARNING")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "id": item["id"],
        "query": item["query"],
        "query_time": item.get("query_time", ""),
        "predicted": predicted,
        "answer": item.get("answer", ""),
        "answer_aliases": item.get("answer_aliases", []),
        "elapsed_ms": round(elapsed_ms, 1),
        "error": error,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Entry point."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    parser = argparse.ArgumentParser(
        description="Run Think-on-Graph QA on CRAG movie questions"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(
            Path(__file__).parent.parent / "dataset" / "crag_movie_tiny.jsonl.bz2"
        ),
        help="Input dataset path (.jsonl or .jsonl.bz2)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSONL file for predictions (default: stdout only)",
    )
    parser.add_argument(
        "--progress",
        type=str,
        default="",
        help="JSON file for progress tracking / resumption",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockGraphBackend (no graph database required)",
    )
    parser.add_argument(
        "--db-uri",
        type=str,
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    )
    parser.add_argument(
        "--db-user",
        type=str,
        default=os.getenv("NEO4J_USER", "neo4j"),
    )
    parser.add_argument(
        "--db-password",
        type=str,
        default=os.getenv("NEO4J_PASSWORD", "password"),
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="movie",
        help="Knowledge domain hint (default: movie)",
    )
    parser.add_argument(
        "--routes",
        type=int,
        default=3,
        help="Number of solving routes (default: 3)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=30,
        help="ToG traversal width (default: 30)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="ToG traversal depth (default: 3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel async workers (default: 1)",
    )
    parser.add_argument(
        "--prefix",
        type=int,
        default=0,
        help="First dataset item index to process (default: 0)",
    )
    parser.add_argument(
        "--postfix",
        type=int,
        default=None,
        help="Exclusive upper bound on dataset items (default: all)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip post-hoc correctness evaluation",
    )
    parser.add_argument(
        "--reset-progress",
        action="store_true",
        help="Delete any existing progress file and re-process all questions",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)

    if args.log_level == "DEBUG":
        import litellm
        litellm.set_verbose = True

    # ------------------------------------------------------------------
    # LLM configuration check
    # ------------------------------------------------------------------
    # Any OpenAI-compatible endpoint is configured via API_BASE + API_KEY.
    # Without API_BASE, the session falls back to direct OpenAI, which
    # requires a valid OPENAI_API_KEY.
    if not args.mock and not os.getenv("API_BASE"):
        log_progress(
            "WARNING: API_BASE is not set. The session will attempt to use the "
            "OpenAI API directly. Set API_BASE (and API_KEY / MODEL_NAME) to "
            "point to your LLM endpoint, or pass --mock for local testing.",
            level="WARNING",
        )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        log_progress(f"Dataset not found: {dataset_path}", level="ERROR")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Progress / resumption
    # ------------------------------------------------------------------
    progress_path = args.progress or ""
    if progress_path and args.reset_progress and Path(progress_path).exists():
        Path(progress_path).unlink()
        log_progress("Progress file deleted; starting fresh.")
    progress = QAProgressLogger(progress_path) if progress_path else None
    if progress:
        progress.load()
        skipped = progress.num_processed
        if skipped:
            log_progress(f"Resuming: {skipped} questions already processed.")

    skip_ids = progress.processed_ids if progress else set()

    # ------------------------------------------------------------------
    # Backend & sessions
    # ------------------------------------------------------------------
    backend = create_backend(
        backend_type="mock" if args.mock else "neo4j",
        neo4j_uri=args.db_uri,
        neo4j_user=args.db_user,
        neo4j_password=args.db_password,
    )

    # Main session — uses API_BASE / API_KEY / MODEL_NAME
    session, model_id = create_session_from_env()
    log_progress(f"Using model: {model_id}, API base: {os.getenv('API_BASE') or '(default)'}")

    # Eval session — uses EVAL_* env vars if set, otherwise reuses main session
    eval_session, _ = (
        create_session_from_env(default_model=model_id, env_prefix="EVAL_")
        if os.getenv("EVAL_API_BASE")
        else (session, model_id)
    )

    emb_client = create_emb_client_optional()

    # ------------------------------------------------------------------
    # Output file
    # ------------------------------------------------------------------
    output_fh = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_fh = open(out_path, "a", encoding="utf-8")

    results: list = []

    async def _process(item: Dict[str, Any]):
        result = await process_question(
            item,
            backend=backend,
            session=session,
            eval_session=eval_session,
            emb_client=emb_client,
            domain=args.domain,
            num_routes=args.routes,
            width=args.width,
            depth=args.depth,
        )
        # Emit immediately
        line = json.dumps(result, ensure_ascii=False, default=str)
        print(line)
        if output_fh:
            output_fh.write(line + "\n")
            output_fh.flush()
        # Update progress
        if progress:
            progress.add_result(item["id"], result)
            progress.save()
        return result

    # ------------------------------------------------------------------
    # Run via loader worker pool
    # ------------------------------------------------------------------
    try:
        loader = MovieDatasetLoader(
            dataset_path=str(dataset_path),
            num_workers=args.workers,
            prefix=args.prefix,
            postfix=args.postfix,
        )
        results = await loader.run(
            process_fn=_process,
            id_key="id",
            skip_ids=skip_ids,
        )
    finally:
        await backend.close()
        if output_fh:
            output_fh.close()

    log_progress(f"Done. {len(results)} questions answered.")

    # ------------------------------------------------------------------
    # Post-hoc evaluation
    # ------------------------------------------------------------------
    if results and not args.no_eval:
        log_progress("Running correctness evaluation...")
        evaluated = await evaluate_predictions(
            session=eval_session,
            predictions=results,
            query_key="query",
            answer_key="predicted",
            gold_key="answer_aliases",
        )
        n_correct = sum(1 for r in evaluated if r.get("correct"))
        accuracy = n_correct / len(evaluated) if evaluated else 0.0
        log_progress(
            f"Accuracy: {n_correct}/{len(evaluated)} = {accuracy:.1%}"
        )

    # ------------------------------------------------------------------
    # Update progress metadata
    # ------------------------------------------------------------------
    if progress:
        progress.update_meta(total_answered=len(results))
        progress.save()


if __name__ == "__main__":
    asyncio.run(main())
