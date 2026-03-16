#!/usr/bin/env python3
"""Evaluate QA results using LLM-based judgement.

Reads a QA results JSONL file (produced by ``run_qa.py``), evaluates each
prediction against the ground-truth answer using a combination of exact
match, fuzzy match, and LLM judgement, then outputs CRAG-style metrics.

Configuration is driven by the same environment variables as the other
scripts:

.. code-block:: bash

    # Optional — separate eval model (defaults to main API_BASE / MODEL_NAME)
    export EVAL_API_BASE=https://your-rits-endpoint/v1
    export EVAL_API_KEY=your-api-key
    export EVAL_MODEL_NAME=meta-llama/llama-3-70b-instruct

Use ``--mock`` to skip LLM calls and evaluate with fuzzy match only
(useful for local testing without a live LLM endpoint).

Example::

    python run_eval.py \\
        --input ../output/qa_results.jsonl \\
        --output ../output/eval_results.json \\
        --metrics ../output/eval_metrics.json

Output JSON format (``--metrics``)::

    {
        "total": 100,
        "n_correct": 72,
        "n_miss": 5,
        "n_hallucination": 23,
        "accuracy": 72.0,
        "score": 49.0,
        "hallucination": 23.0,
        "missing": 5.0,
        "eval_model": "meta-llama/llama-3-70b-instruct"
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from mellea_contribs.kg.utils import (
    create_session_from_env,
    evaluate_predictions,
    load_jsonl,
    log_progress,
    setup_logging,
)


def compute_crag_metrics(
    evaluated: List[Dict[str, Any]],
    eval_model: str = "",
) -> Dict[str, Any]:
    """Compute CRAG-style evaluation metrics.

    Args:
        evaluated: List of evaluated result dicts (each has a ``"correct"`` field).
        eval_model: Model name used for evaluation (for logging).

    Returns:
        Dict with total, n_correct, n_miss, n_hallucination, accuracy, score,
        hallucination, missing, and eval_model fields.
    """
    n = len(evaluated)
    if n == 0:
        return {"total": 0, "eval_model": eval_model}

    n_correct = sum(1 for r in evaluated if r.get("correct"))
    # "I don't know" responses count as missing (not hallucination)
    n_miss = sum(
        1 for r in evaluated
        if "i don't know" in str(r.get("predicted", "")).lower()
        and not r.get("correct")
    )
    n_hallucination = n - n_correct - n_miss

    accuracy = (n_correct / n) * 100.0
    # CRAG score formula: penalises hallucination more than missing answers
    crag_score = ((2 * n_correct + n_miss) / n - 1) * 100.0

    return {
        "total": n,
        "n_correct": n_correct,
        "n_miss": n_miss,
        "n_hallucination": n_hallucination,
        "accuracy": round(accuracy, 2),
        "score": round(crag_score, 2),
        "hallucination": round((n_hallucination / n) * 100.0, 2),
        "missing": round((n_miss / n) * 100.0, 2),
        "eval_model": eval_model,
    }


async def main() -> None:
    """Entry point."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    parser = argparse.ArgumentParser(
        description="Evaluate QA results with LLM-based judgement"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with QA results (from run_qa.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSONL file with per-item evaluation scores added",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Output JSON file for aggregate CRAG metrics",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Skip LLM evaluation; use fuzzy match only",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(log_level=args.log_level)

    # ------------------------------------------------------------------
    # Load QA results
    # ------------------------------------------------------------------
    input_path = Path(args.input)
    if not input_path.exists():
        log_progress(f"Input file not found: {input_path}", level="ERROR")
        sys.exit(1)

    results: List[Dict[str, Any]] = list(load_jsonl(input_path))
    log_progress(f"Loaded {len(results)} results from {input_path}")

    if not results:
        log_progress("No results to evaluate.", level="WARNING")
        sys.exit(0)

    # ------------------------------------------------------------------
    # LLM-based evaluation
    # ------------------------------------------------------------------
    eval_model = ""

    if args.mock:
        log_progress("--mock: skipping LLM evaluation; using fuzzy match only.")
        # evaluate_predictions handles the None session path gracefully via
        # exact/fuzzy match before the LLM branch — pass a dummy session.
        session = None
        evaluated = await evaluate_predictions(
            session=session,
            predictions=results,
            query_key="query",
            answer_key="predicted",
            gold_key="answer_aliases",
        )
    else:
        # Use EVAL_* vars if set, otherwise fall back to main session vars.
        if os.getenv("EVAL_API_BASE"):
            session, eval_model = create_session_from_env(env_prefix="EVAL_")
        else:
            session, eval_model = create_session_from_env()
        log_progress(f"Evaluating with model: {eval_model}")

        evaluated = await evaluate_predictions(
            session=session,
            predictions=results,
            query_key="query",
            answer_key="predicted",
            gold_key="answer_aliases",
        )

    # ------------------------------------------------------------------
    # Compute and display metrics
    # ------------------------------------------------------------------
    metrics = compute_crag_metrics(evaluated, eval_model=eval_model)

    log_progress("=" * 50)
    log_progress("Evaluation Results")
    log_progress("=" * 50)
    log_progress(f"Total questions : {metrics['total']}")
    log_progress(f"Correct         : {metrics['n_correct']}")
    log_progress(f"Hallucination   : {metrics['n_hallucination']}")
    log_progress(f"Missing         : {metrics['n_miss']}")
    log_progress(f"Accuracy        : {metrics['accuracy']:.1f}%")
    log_progress(f"CRAG Score      : {metrics['score']:.1f}")
    log_progress("=" * 50)

    # ------------------------------------------------------------------
    # Save annotated results
    # ------------------------------------------------------------------
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            for item in evaluated:
                fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        log_progress(f"Annotated results saved to {out_path}")

    # ------------------------------------------------------------------
    # Save aggregate metrics
    # ------------------------------------------------------------------
    if args.metrics:
        metrics_path = Path(args.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        log_progress(f"Metrics saved to {metrics_path}")

    # Always print metrics to stdout as JSON
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
