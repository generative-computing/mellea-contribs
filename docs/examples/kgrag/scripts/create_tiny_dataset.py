#!/usr/bin/env python3
"""Create a tiny JSONL dataset with ~5 synthetic movie Q&A pairs.

Creates a minimal JSONL file with 5 movie Q&A pairs for quick sanity-check runs.
This is a subset of the demo dataset for testing with minimal LLM calls.

Example::

    python create_tiny_dataset.py --output /tmp/tiny.jsonl
"""

import argparse
import json
from pathlib import Path


# Minimal set of movie Q&A pairs for quick tests
TINY_QA_PAIRS = [
    {
        "question": "Who directed Avatar?",
        "answer": "James Cameron",
        "domain": "movie"
    },
    {
        "question": "What year was Titanic released?",
        "answer": "1997",
        "domain": "movie"
    },
    {
        "question": "Who directed The Matrix?",
        "answer": "The Wachowskis",
        "domain": "movie"
    },
    {
        "question": "What studio produced Inception?",
        "answer": "Warner Bros",
        "domain": "movie"
    },
    {
        "question": "In which year was Parasite released?",
        "answer": "2019",
        "domain": "movie"
    },
]


def main():
    """Create tiny dataset and write to JSONL file."""
    parser = argparse.ArgumentParser(
        description="Create a tiny JSONL dataset with 5 movie Q&A pairs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tiny_dataset.jsonl",
        help="Output file path (default: data/tiny_dataset.jsonl)",
    )
    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write tiny Q&A pairs to JSONL
    with open(output_path, "w") as f:
        for pair in TINY_QA_PAIRS:
            f.write(json.dumps(pair) + "\n")

    print(f"Created tiny dataset with {len(TINY_QA_PAIRS)} Q&A pairs")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
