"""JSONL and data processing utilities.

Provides reusable functions for reading/writing JSONL files, batch processing,
and dataset manipulation.
"""

import json
import random
import sys
from pathlib import Path
from typing import Iterator, List, Dict, Any, Callable, Optional


def load_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Load JSONL file and yield each line as a dictionary.

    Args:
        path: Path to JSONL file.

    Yields:
        Dictionary from each JSON line.

    Raises:
        FileNotFoundError: If file does not exist.
        json.JSONDecodeError: If line is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Line {line_num}] JSON decode error: {e}", file=sys.stderr)
                raise


def save_jsonl(data: List[Dict[str, Any]], path: Path) -> None:
    """Save list of dictionaries as JSONL file.

    Args:
        data: List of dictionaries to save.
        path: Path to output JSONL file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def append_jsonl(item: Dict[str, Any], path: Path) -> None:
    """Append a single dictionary to JSONL file.

    Args:
        item: Dictionary to append.
        path: Path to JSONL file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a") as f:
        f.write(json.dumps(item) + "\n")


def batch_iterator(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """Iterate through items in batches.

    Args:
        items: List of items to batch.
        batch_size: Size of each batch.

    Yields:
        Lists of items, each of size batch_size (last batch may be smaller).
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def stream_batch_process(
    input_path: Path,
    output_path: Path,
    process_fn: Callable,
    batch_size: int = 1,
) -> int:
    """Process JSONL file in batches and write results.

    Args:
        input_path: Path to input JSONL file.
        output_path: Path to output JSONL file.
        process_fn: Function that takes a list of items and returns processed list.
        batch_size: Number of items to process at once (default: 1).

    Returns:
        Number of items processed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    batch = []

    with open(output_path, "w") as out_f:
        try:
            for item in load_jsonl(input_path):
                batch.append(item)
                count += 1

                if len(batch) >= batch_size:
                    # Process batch
                    processed = process_fn(batch)
                    for result in processed:
                        out_f.write(json.dumps(result) + "\n")
                    batch = []

            # Process remaining items
            if batch:
                processed = process_fn(batch)
                for result in processed:
                    out_f.write(json.dumps(result) + "\n")

        except Exception as e:
            print(f"Error during batch processing: {e}", file=sys.stderr)
            raise

    return count


def truncate_jsonl(
    input_path: Path, output_path: Path, max_lines: int
) -> int:
    """Truncate JSONL file to specified number of lines.

    Args:
        input_path: Path to input JSONL file.
        output_path: Path to output truncated JSONL file.
        max_lines: Maximum number of lines to keep.

    Returns:
        Number of lines written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as out_f:
        for item in load_jsonl(input_path):
            if count >= max_lines:
                break
            out_f.write(json.dumps(item) + "\n")
            count += 1

    return count


def shuffle_jsonl(input_path: Path, output_path: Path) -> int:
    """Shuffle JSONL file randomly.

    Args:
        input_path: Path to input JSONL file.
        output_path: Path to output shuffled JSONL file.

    Returns:
        Number of lines written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all items
    items = list(load_jsonl(input_path))

    # Shuffle
    random.shuffle(items)

    # Write
    with open(output_path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    return len(items)


def validate_jsonl_schema(
    path: Path, required_fields: List[str]
) -> tuple[bool, List[str]]:
    """Validate that all items in JSONL have required fields.

    Args:
        path: Path to JSONL file.
        required_fields: List of field names that must be present.

    Returns:
        Tuple of (is_valid, error_messages).
    """
    errors = []

    try:
        for line_num, item in enumerate(load_jsonl(path), 1):
            for field in required_fields:
                if field not in item:
                    errors.append(f"Line {line_num}: Missing field '{field}'")
    except Exception as e:
        errors.append(f"Error validating file: {e}")

    return len(errors) == 0, errors
