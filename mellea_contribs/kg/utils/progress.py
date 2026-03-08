"""Progress tracking and logging utilities.

Provides functions for logging, progress tracking, and structured output.
"""

import json
import logging
import sys
from typing import Union, Optional

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
