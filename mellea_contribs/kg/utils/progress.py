"""Progress tracking and logging utilities.

Provides functions for logging, progress tracking, and structured output.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional, Set, Union

from pydantic import BaseModel

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging for the application.

    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", default: "INFO").
        log_file: Optional file path to write logs to.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger("mellea_contribs.kg")
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def log_progress(msg: str, level: str = "INFO") -> None:
    """Log a progress message to stderr.

    Args:
        msg: Message to log.
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", default: "INFO").
    """
    logger = logging.getLogger("mellea_contribs.kg")
    level_func = getattr(logger, level.lower(), logger.info)
    level_func(msg)


def output_json(obj: BaseModel) -> None:
    """Output a Pydantic model as JSON to stdout.

    Args:
        obj: Pydantic model instance to output.
    """
    print(json.dumps(obj.model_dump()))


def print_stats(
    stats: BaseModel, indent: int = 0, to_stderr: bool = True
) -> None:
    """Pretty-print statistics to stderr or stdout.

    Args:
        stats: Statistics object (QAStats, UpdateStats, EmbeddingStats, etc.).
        indent: Number of spaces to indent (default: 0).
        to_stderr: Print to stderr if True, stdout if False (default: True).
    """
    output = sys.stderr if to_stderr else sys.stdout
    prefix = " " * indent

    # Get all fields from stats object
    data = stats.model_dump()

    for key, value in data.items():
        # Format key (snake_case to Title Case)
        display_key = key.replace("_", " ").title()

        # Format value
        if isinstance(value, float):
            display_value = f"{value:.2f}"
        elif isinstance(value, list):
            display_value = ", ".join(str(v) for v in value)
        else:
            display_value = str(value)

        print(f"{prefix}{display_key}: {display_value}", file=output)


class ProgressTracker:
    """Progress tracker with optional tqdm integration.

    If tqdm is available, uses progress bar; otherwise prints text updates.
    """

    def __init__(self, total: int, desc: str = "Processing", use_tqdm: bool = True):
        """Initialize progress tracker.

        Args:
            total: Total number of items to process.
            desc: Description of progress (default: "Processing").
            use_tqdm: Use tqdm if available (default: True).
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self.use_tqdm = use_tqdm and tqdm is not None

        if self.use_tqdm:
            self.pbar = tqdm(total=total, desc=desc)
        else:
            self.pbar = None

    def update(self, n: int = 1) -> None:
        """Update progress by n items.

        Args:
            n: Number of items to add to progress (default: 1).
        """
        self.current += n

        if self.use_tqdm and self.pbar:
            self.pbar.update(n)
        else:
            # Print text update
            percent = (self.current / self.total) * 100
            print(
                f"{self.desc}: {self.current}/{self.total} ({percent:.1f}%)",
                file=sys.stderr,
            )

    def close(self) -> None:
        """Close the progress tracker."""
        if self.use_tqdm and self.pbar:
            self.pbar.close()


class BaseProgressLogger:
    """JSON-file-backed progress logger with resumption support.

    Persists a set of processed item IDs and arbitrary key-value metadata
    to a JSON file so that long-running pipelines can resume after
    interruption.

    Usage::

        logger = BaseProgressLogger("progress.json")
        logger.load()
        for item in items:
            if logger.is_processed(item["id"]):
                continue
            result = process(item)
            logger.mark_processed(item["id"])
            logger.add_stat(result)
            logger.save()
    """

    def __init__(self, progress_path: str) -> None:
        """Initialise the logger.

        Args:
            progress_path: Path to the JSON file used for persistence.
        """
        self._path = progress_path
        self._processed: Set[str] = set()
        self._stats: list = []
        self._meta: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load existing progress from disk (no-op when file is absent)."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._processed = set(data.get("processed", []))
            self._stats = data.get("stats", [])
            self._meta = data.get("meta", {})
        except (json.JSONDecodeError, OSError):
            pass

    def save(self, retries: int = 3) -> None:
        """Save progress to disk.

        Args:
            retries: Number of write attempts before giving up.
        """
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        data = {
            "processed": list(self._processed),
            "stats": self._stats,
            "meta": self._meta,
        }
        for attempt in range(retries):
            try:
                tmp = self._path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, default=str)
                os.replace(tmp, self._path)
                return
            except OSError:
                if attempt < retries - 1:
                    time.sleep(0.1)

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def is_processed(self, item_id: str) -> bool:
        """Check whether an item has already been processed.

        Args:
            item_id: Unique identifier for the item.

        Returns:
            True if the item is in the processed set.
        """
        return item_id in self._processed

    def mark_processed(self, item_id: str) -> None:
        """Mark an item as processed.

        Args:
            item_id: Unique identifier for the item.
        """
        self._processed.add(item_id)

    def add_stat(self, stat: Any) -> None:
        """Append a result/stat entry.

        Args:
            stat: Any JSON-serialisable value.
        """
        if hasattr(stat, "model_dump"):
            self._stats.append(stat.model_dump())
        else:
            self._stats.append(stat)

    def update_meta(self, **kwargs: Any) -> None:
        """Update key-value metadata entries.

        Args:
            **kwargs: Key-value pairs to store in the metadata dict.
        """
        self._meta.update(kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def processed_ids(self) -> Set[str]:
        """Set of all processed item IDs."""
        return set(self._processed)

    @property
    def stats(self) -> list:
        """List of collected stat entries."""
        return list(self._stats)

    @property
    def meta(self) -> Dict[str, Any]:
        """Metadata dictionary."""
        return dict(self._meta)

    @property
    def num_processed(self) -> int:
        """Number of items marked as processed."""
        return len(self._processed)


class QAProgressLogger(BaseProgressLogger):
    """Progress logger specialised for QA pipeline runs.

    Stores per-question results alongside a processed-question ID set so
    that the run can resume mid-dataset without repeating work.

    The progress file uses the format::

        {
            "processed": ["q_0", "q_3", ...],
            "stats": [{"query": "...", "answer": "...", ...}, ...],
            "meta": {"total": 100, "last_updated": "..."}
        }
    """

    def add_result(self, query_id: str, result: Any) -> None:
        """Record a QA result and mark the query as processed.

        Args:
            query_id: Unique identifier for the query (used for resumption).
            result: QAResult, dict, or any JSON-serialisable value.
        """
        self.add_stat(result)
        self.mark_processed(query_id)

    @property
    def processed_queries(self) -> Set[str]:
        """Set of query IDs that have been answered."""
        return self.processed_ids
