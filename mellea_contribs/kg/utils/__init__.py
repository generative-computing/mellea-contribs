"""KG utility modules for JSONL I/O, session management, progress tracking, and evaluation.

This package provides reusable utilities extracted from run scripts:
- data_utils: JSONL reading/writing, batch processing
- session_manager: Mellea session and backend creation
- progress: Logging, progress tracking, structured output
- eval_utils: Evaluation metrics and result aggregation
"""

from .data_utils import (
    load_jsonl,
    save_jsonl,
    append_jsonl,
    batch_iterator,
    stream_batch_process,
    truncate_jsonl,
    shuffle_jsonl,
    validate_jsonl_schema,
)
from .session_manager import (
    create_session,
    create_backend,
    MelleaResourceManager,
)
from .progress import (
    setup_logging,
    log_progress,
    output_json,
    print_stats,
    ProgressTracker,
)
from .eval_utils import (
    exact_match,
    fuzzy_match,
    mean_reciprocal_rank,
    precision,
    recall,
    f1_score,
    aggregate_qa_results,
    aggregate_update_results,
)

__all__ = [
    # data_utils
    "load_jsonl",
    "save_jsonl",
    "append_jsonl",
    "batch_iterator",
    "stream_batch_process",
    "truncate_jsonl",
    "shuffle_jsonl",
    "validate_jsonl_schema",
    # session_manager
    "create_session",
    "create_backend",
    "MelleaResourceManager",
    # progress
    "setup_logging",
    "log_progress",
    "output_json",
    "print_stats",
    "ProgressTracker",
    # eval_utils
    "exact_match",
    "fuzzy_match",
    "mean_reciprocal_rank",
    "precision",
    "recall",
    "f1_score",
    "aggregate_qa_results",
    "aggregate_update_results",
]
