#!/usr/bin/env python3
"""Truncate an existing JSONL dataset to N examples.

Reads a JSONL dataset file and outputs a truncated version with at most
N examples. Useful for creating dev subsets from larger datasets.

Example::

    python create_truncated_dataset.py --input data/full.jsonl --output data/dev.jsonl --max-examples 10
"""

import argparse
import random
from pathlib import Path

from mellea_contribs.kg.utils import load_jsonl, save_jsonl


def main():
    """Truncate dataset and write to JSONL file."""
    parser = argparse.ArgumentParser(
        description="Truncate a JSONL dataset to N examples"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Maximum number of examples to keep (default: 10)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle examples before truncating",
    )
    args = parser.parse_args()

    # Read input dataset using utility
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    try:
        examples = list(load_jsonl(input_path))
    except Exception as e:
        print(f"Error: Failed to read input file: {e}")
        return

    print(f"Loaded {len(examples)} examples from {input_path}")

    # Shuffle if requested
    if args.shuffle:
        random.shuffle(examples)
        print(f"Shuffled examples")

    # Truncate to max examples
    truncated = examples[: args.max_examples]
    print(f"Truncated to {len(truncated)} examples")

    # Write truncated dataset to JSONL using utility
    output_path = Path(args.output)
    save_jsonl(truncated, output_path)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
