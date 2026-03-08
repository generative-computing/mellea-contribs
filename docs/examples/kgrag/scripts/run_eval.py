#!/usr/bin/env python3
"""Evaluate QA results and compute metrics.

Reads QA output JSONL file and computes evaluation metrics including exact match,
fuzzy match (via rapidfuzz), and mean reciprocal rank.

Example::

    python run_eval.py --input /tmp/qa_results.jsonl --output /tmp/metrics.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from mellea_contribs.kg.qa_models import QAStats
from mellea_contribs.kg.utils import (
    load_jsonl,
    log_progress,
    output_json,
    print_stats,
    mean_reciprocal_rank,
)


def eval_qa_results(
    input_path: Path,
    exact_match: bool = True,
    fuzzy_threshold: float = 0.8,
) -> QAStats:
    """Evaluate QA results from JSONL file."""
    stats = QAStats()
    results = []

    try:
        for result in load_jsonl(input_path):
            results.append(result)

            error = result.get("error")

            # Skip if there was an error
            if error:
                stats.failed_answers += 1
                continue

            stats.successful_answers += 1

    except Exception as e:
        log_progress(f"ERROR: Failed to read input file: {e}")
        return stats

    # Compute aggregate stats
    stats.total_questions = len(results)

    # Compute confidence-based metrics if available
    confidences = [
        r.get("confidence", 0.0) for r in results if "confidence" in r
    ]
    if confidences:
        stats.average_confidence = sum(confidences) / len(confidences)

    # Compute processing time metrics if available
    times = [
        r.get("processing_time_ms", 0.0)
        for r in results
        if "processing_time_ms" in r
    ]
    if times:
        stats.average_processing_time_ms = sum(times) / len(times)
        stats.min_processing_time_ms = min(times)
        stats.max_processing_time_ms = max(times)
        stats.total_time_ms = sum(times)

    # Compute MRR using utility
    stats.mean_reciprocal_rank = mean_reciprocal_rank(results)

    # Collect models used
    models = set(
        r.get("model_used", "unknown") for r in results if "model_used" in r
    )
    stats.models_used = list(models)

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate QA results and compute metrics"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with QA results",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for QAStats (optional)",
    )
    parser.add_argument(
        "--exact-match",
        action="store_true",
        default=True,
        help="Compute exact match metric",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.8,
        help="Fuzzy match threshold (0-1, default: 0.8)",
    )
    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        log_progress(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Evaluate results
    try:
        stats = eval_qa_results(
            input_path=input_path,
            exact_match=args.exact_match,
            fuzzy_threshold=args.fuzzy_threshold,
        )
    except Exception as e:
        log_progress(f"ERROR: Evaluation failed: {e}")
        sys.exit(1)

    # Print summary table using utility
    log_progress("\n=== Evaluation Metrics ===")
    print_stats(stats, to_stderr=True)

    # Output stats to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(stats.model_dump(), f, indent=2)
        log_progress(f"Stats saved to: {output_path}")

    # Write stats to stdout as JSON
    output_json(stats)


if __name__ == "__main__":
    main()
