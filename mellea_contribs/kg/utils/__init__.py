"""KG utility modules for JSONL I/O, session management, progress tracking, and evaluation.

This package provides reusable utilities extracted from run scripts:
- data_utils: JSONL reading/writing, batch processing
- session_manager: Mellea session and backend creation
- progress: Logging, progress tracking, structured output
- eval_utils: Evaluation metrics and result aggregation
"""

from .data_utils import (
    BaseDatasetLoader,
    append_jsonl,
    batch_iterator,
    load_jsonl,
    save_jsonl,
    shuffle_jsonl,
    stream_batch_process,
    truncate_jsonl,
    validate_jsonl_schema,
)
from .eval_utils import (
    aggregate_qa_results,
    aggregate_update_results,
    evaluate_predictions,
    exact_match,
    f1_score,
    fuzzy_match,
    mean_reciprocal_rank,
    precision,
    recall,
)
from .progress import (
    BaseProgressLogger,
    ProgressTracker,
    QAProgressLogger,
    log_progress,
    output_json,
    print_stats,
    setup_logging,
)
from .session_manager import (
    MelleaResourceManager,
    create_backend,
    create_embedding_client,
    create_openai_session,
    create_session,
    create_session_from_env,
    generate_embeddings,
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
    "BaseDatasetLoader",
    # session_manager
    "create_session",
    "create_openai_session",
    "create_session_from_env",
    "create_backend",
    "create_embedding_client",
    "generate_embeddings",
    "MelleaResourceManager",
    # progress
    "setup_logging",
    "log_progress",
    "output_json",
    "print_stats",
    "ProgressTracker",
    "BaseProgressLogger",
    "QAProgressLogger",
    # eval_utils
    "exact_match",
    "fuzzy_match",
    "mean_reciprocal_rank",
    "precision",
    "recall",
    "f1_score",
    "aggregate_qa_results",
    "aggregate_update_results",
    "evaluate_predictions",
]
