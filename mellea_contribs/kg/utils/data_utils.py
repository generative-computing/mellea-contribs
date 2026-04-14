"""JSONL and data processing utilities.

Provides reusable functions for reading/writing JSONL files, batch processing,
and dataset manipulation.
"""

import asyncio
import bz2
import json
import random
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Set


def load_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Load JSONL file and yield each line as a dictionary.

    Supports both plain text and bz2-compressed JSONL files.

    Args:
        path: Path to JSONL file (plain or .bz2).

    Yields:
        Dictionary from each JSON line.

    Raises:
        FileNotFoundError: If file does not exist.
        json.JSONDecodeError: If line is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if str(path).endswith('.bz2'):
        f = bz2.open(path, "rt", encoding="utf-8")
    else:
        f = open(path, "r")

    try:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Line {line_num}] JSON decode error: {e}", file=sys.stderr)
                raise
    finally:
        f.close()


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


class BaseDatasetLoader(ABC):
    """Abstract base class for async dataset loaders with worker-pool support.

    Subclasses implement :meth:`iter_items` to yield raw dataset records.
    :meth:`run` feeds those records through a configurable number of async
    workers, skipping IDs that appear in ``skip_ids``.

    Usage::

        class MyLoader(BaseDatasetLoader):
            def iter_items(self):
                for item in load_jsonl(self.dataset_path):
                    yield item

        loader = MyLoader(dataset_path="data.jsonl", num_workers=4)
        results = await loader.run(
            process_fn=my_async_fn,
            id_key="id",
            skip_ids=already_done,
        )
    """

    def __init__(
        self,
        dataset_path: str,
        num_workers: int = 1,
        queue_size: int = 100,
    ) -> None:
        """Initialise the loader.

        Args:
            dataset_path: Path to the dataset file.
            num_workers: Number of parallel async workers (default: ``1``).
            queue_size: Internal asyncio queue capacity (default: ``100``).
        """
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.queue_size = queue_size

    @abstractmethod
    def iter_items(self) -> Generator[Dict[str, Any], None, None]:
        """Yield raw dataset items one by one.

        Subclasses must implement this method.  It is called synchronously
        from the producer coroutine inside :meth:`run`.

        Yields:
            Dict representing a single dataset record.
        """

    async def run(
        self,
        process_fn: Callable,
        id_key: str = "id",
        skip_ids: Optional[Set[str]] = None,
        on_result: Optional[Callable] = None,
    ) -> List[Any]:
        """Process all items through an async worker pool.

        Args:
            process_fn: ``async (item) -> result`` coroutine called for each
                item.  Should return *None* to discard results.
            id_key: Key in each item dict used as the unique ID for
                ``skip_ids`` matching (default: ``"id"``).
            skip_ids: Set of item IDs to skip (for resumption).
            on_result: Optional async or sync callback ``(item_id, result)``
                invoked after each item is processed successfully.

        Returns:
            List of non-None results collected from ``process_fn``.
        """
        skip_ids = skip_ids or set()
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size)
        results: List[Any] = []
        _sentinel = object()

        async def _producer() -> None:
            for item in self.iter_items():
                item_id = str(item.get(id_key, ""))
                if item_id and item_id in skip_ids:
                    continue
                await queue.put(item)
            # Send one sentinel per worker to signal end-of-stream
            for _ in range(self.num_workers):
                await queue.put(_sentinel)

        async def _worker() -> None:
            while True:
                item = await queue.get()
                if item is _sentinel:
                    queue.task_done()
                    break
                try:
                    result = await process_fn(item)
                    if result is not None:
                        results.append(result)
                        if on_result is not None:
                            item_id = str(item.get(id_key, ""))
                            if asyncio.iscoroutinefunction(on_result):
                                await on_result(item_id, result)
                            else:
                                on_result(item_id, result)
                except Exception as exc:
                    print(
                        f"Worker error on item {item.get(id_key, '?')}: {exc}",
                        file=sys.stderr,
                    )
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(_worker()) for _ in range(self.num_workers)]
        await asyncio.gather(_producer(), *workers)
        return results
