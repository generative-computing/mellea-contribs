#!/usr/bin/env python3
"""Run QA on questions via orchestrate_qa_retrieval.

Reads questions from a JSONL file and runs QA using the orchestration function.
Produces QAResult written to output JSONL file. Optionally prints running accuracy
if ground truth answers are present.

Input format: Each line is JSON with 'question' field and optional 'answer' field.
Output format: QAResult fields as JSON per line.

Example::

    python run_qa.py --input data/questions.jsonl --output /tmp/qa_results.jsonl --mock
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

try:
    from mellea import start_session
except ImportError:
    print("Error: mellea not installed. Run: pip install mellea[litellm]")
    sys.exit(1)

from mellea_contribs.kg.graph_dbs.base import GraphBackend
from mellea_contribs.kg.graph_dbs.mock import MockGraphBackend

try:
    from mellea_contribs.kg.graph_dbs.neo4j import Neo4jBackend
except ImportError:
    Neo4jBackend = None

from mellea_contribs.kg.kgrag import orchestrate_qa_retrieval
from mellea_contribs.kg.qa_models import QAResult, QAStats


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


async def run_qa_on_questions(
    input_path: Path,
    backend: GraphBackend,
    session,
    model: str,
    domain: str,
    num_routes: int,
    format_style: str,
) -> tuple[list[QAResult], QAStats]:
    """Run QA on questions from JSONL file."""
    qa_results = []
    stats = QAStats()
    times = []
    confidences = []
    matches = 0

    try:
        with open(input_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    question = item.get("question", "")
                    ground_truth = item.get("answer")

                    if not question:
                        print(
                            f"[{line_num}] WARNING: Empty question",
                            file=sys.stderr
                        )
                        continue

                    q_start = time.time()

                    try:
                        # Call orchestrate_qa_retrieval
                        answer = await orchestrate_qa_retrieval(
                            session=session,
                            backend=backend,
                            query=question,
                            query_time="",
                            domain=domain,
                            num_routes=num_routes,
                            hints="",
                        )

                        q_elapsed = time.time() - q_start

                        # Create QAResult
                        result = QAResult(
                            question=question,
                            answer=answer,
                            confidence=0.7,  # Default confidence
                            processing_time_ms=q_elapsed * 1000,
                            model_used=model,
                        )
                        qa_results.append(result)
                        times.append(q_elapsed)
                        confidences.append(result.confidence)
                        stats.successful_answers += 1

                        # Check if answer matches ground truth
                        if ground_truth and answer.lower() == ground_truth.lower():
                            matches += 1
                            print(
                                f"[{line_num}] ✓ {question} ({q_elapsed:.2f}s)",
                                file=sys.stderr
                            )
                        else:
                            print(
                                f"[{line_num}] ? {question} ({q_elapsed:.2f}s)",
                                file=sys.stderr
                            )

                    except Exception as inner_e:
                        print(f"[{line_num}] ERROR in QA: {inner_e}", file=sys.stderr)
                        result = QAResult(
                            question=question,
                            answer="",
                            confidence=0.0,
                            error=str(inner_e),
                            processing_time_ms=(time.time() - q_start) * 1000,
                            model_used=model,
                        )
                        qa_results.append(result)
                        stats.failed_answers += 1

                except json.JSONDecodeError as e:
                    print(f"[{line_num}] ERROR: Failed to parse JSON: {e}", file=sys.stderr)
                    stats.failed_answers += 1

    except IOError as e:
        print(f"ERROR: Failed to read input file: {e}", file=sys.stderr)
        return qa_results, stats

    # Compute aggregate stats
    stats.total_questions = len(qa_results)
    if times:
        stats.average_processing_time_ms = sum(times) / len(times) * 1000
        stats.min_processing_time_ms = min(times) * 1000
        stats.max_processing_time_ms = max(times) * 1000
    if confidences:
        stats.average_confidence = sum(confidences) / len(confidences)
    if ground_truth is not None:
        stats.exact_match_count = matches
    stats.models_used = [model]

    return qa_results, stats


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run QA on questions via orchestrate_qa_retrieval"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file path (each line: {\"question\": \"...\", \"answer\": \"...\"})",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSONL file for QAResult (optional)",
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
        help="Domain for QA hints (default: general)",
    )
    parser.add_argument(
        "--routes",
        type=int,
        default=3,
        help="Number of solving routes (default: 3)",
    )
    parser.add_argument(
        "--format-style",
        type=str,
        default="natural",
        help="Result format style (default: natural)",
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
        # Run QA
        qa_results, stats = await run_qa_on_questions(
            input_path=input_path,
            backend=backend,
            session=session,
            model=args.model,
            domain=args.domain,
            num_routes=args.routes,
            format_style=args.format_style,
        )

        # Print summary
        print("\n=== QA Summary ===", file=sys.stderr)
        print(f"Total questions: {stats.total_questions}", file=sys.stderr)
        print(f"Successful: {stats.successful_answers}", file=sys.stderr)
        print(f"Failed: {stats.failed_answers}", file=sys.stderr)
        print(f"Average confidence: {stats.average_confidence:.2f}", file=sys.stderr)
        print(
            f"Average time per question: {stats.average_processing_time_ms:.2f}ms",
            file=sys.stderr
        )
        if stats.exact_match_count > 0:
            accuracy = stats.exact_match_count / stats.total_questions * 100
            print(f"Exact match accuracy: {accuracy:.1f}%", file=sys.stderr)

        # Write results to output file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for result in qa_results:
                    f.write(json.dumps(result.model_dump()) + "\n")
            print(f"\nResults saved to: {output_path}", file=sys.stderr)

        # Write results to stdout as JSONL
        for result in qa_results:
            print(json.dumps(result.model_dump()))

    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
