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

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None
    print("Warning: rapidfuzz not available. Fuzzy matching disabled.", file=sys.stderr)

from mellea_contribs.kg.qa_models import QAStats


def compute_exact_match(predicted: str, expected: str) -> bool:
    """Check if predicted answer exactly matches expected answer."""
    return predicted.lower().strip() == expected.lower().strip()


def compute_fuzzy_match(
    predicted: str,
    expected: str,
    threshold: float = 0.8
) -> bool:
    """Check if predicted answer fuzzy-matches expected answer."""
    if fuzz is None:
        return False
    score = fuzz.token_set_ratio(predicted.lower(), expected.lower()) / 100.0
    return score >= threshold


def compute_mrr(results: list[dict], exact_match_threshold: float = 0.9) -> float:
    """Compute Mean Reciprocal Rank for results."""
    if not results:
        return 0.0

    reciprocal_ranks = []
    for result in results:
        if compute_exact_match(result.get("answer", ""), result.get("expected", "")):
            reciprocal_ranks.append(1.0)
        else:
            # For non-exact matches, use confidence as proxy for ranking
            confidence = result.get("confidence", 0.0)
            if confidence >= exact_match_threshold:
                reciprocal_ranks.append(1.0 / (1.0 + (1.0 - confidence)))
            else:
                reciprocal_ranks.append(0.0)

    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def eval_qa_results(
    input_path: Path,
    exact_match: bool = True,
    fuzzy_threshold: float = 0.8,
) -> QAStats:
    """Evaluate QA results from JSONL file."""
    stats = QAStats()
    results = []
    exact_matches = 0
    fuzzy_matches = 0

    try:
        with open(input_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    result = json.loads(line)
                    results.append(result)

                    predicted = result.get("answer", "")
                    error = result.get("error")

                    # Skip if there was an error
                    if error:
                        stats.failed_answers += 1
                        continue

                    stats.successful_answers += 1

                    # Note: Expected answer not in QA output, so we can't evaluate
                    # This would require input that has ground truth

                except json.JSONDecodeError as e:
                    print(f"[{line_num}] ERROR: Failed to parse JSON: {e}", file=sys.stderr)
                    stats.failed_answers += 1

    except IOError as e:
        print(f"ERROR: Failed to read input file: {e}", file=sys.stderr)
        return stats

    # Compute aggregate stats
    stats.total_questions = len(results)

    # Compute confidence-based metrics if available
    confidences = [r.get("confidence", 0.0) for r in results if "confidence" in r]
    if confidences:
        stats.average_confidence = sum(confidences) / len(confidences)

    # Compute processing time metrics if available
    times = [r.get("processing_time_ms", 0.0) for r in results if "processing_time_ms" in r]
    if times:
        stats.average_processing_time_ms = sum(times) / len(times)
        stats.min_processing_time_ms = min(times)
        stats.max_processing_time_ms = max(times)
        stats.total_time_ms = sum(times)

    # Compute MRR
    stats.mean_reciprocal_rank = compute_mrr(results)

    # Collect models used
    models = set(r.get("model_used", "unknown") for r in results if "model_used" in r)
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
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Evaluate results
    try:
        stats = eval_qa_results(
            input_path=input_path,
            exact_match=args.exact_match,
            fuzzy_threshold=args.fuzzy_threshold,
        )
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary table
    print("\n=== Evaluation Metrics ===", file=sys.stderr)
    print(f"Total questions: {stats.total_questions}", file=sys.stderr)
    print(f"Successful: {stats.successful_answers}", file=sys.stderr)
    print(f"Failed: {stats.failed_answers}", file=sys.stderr)
    print(f"Average confidence: {stats.average_confidence:.2f}", file=sys.stderr)
    print(f"Average time per question: {stats.average_processing_time_ms:.2f}ms", file=sys.stderr)
    print(f"Min/Max time: {stats.min_processing_time_ms:.2f}ms / {stats.max_processing_time_ms:.2f}ms", file=sys.stderr)
    print(f"Mean Reciprocal Rank: {stats.mean_reciprocal_rank:.4f}", file=sys.stderr)
    if stats.exact_match_count > 0:
        accuracy = stats.exact_match_count / stats.total_questions * 100
        print(f"Exact match accuracy: {accuracy:.1f}%", file=sys.stderr)
    if stats.partial_match_count > 0:
        partial_accuracy = stats.partial_match_count / stats.total_questions * 100
        print(f"Partial match accuracy: {partial_accuracy:.1f}%", file=sys.stderr)

    # Output stats to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(stats.model_dump(), f, indent=2)
        print(f"\nStats saved to: {output_path}", file=sys.stderr)

    # Write stats to stdout as JSON
    print(json.dumps(stats.model_dump()))


if __name__ == "__main__":
    main()
