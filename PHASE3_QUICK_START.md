# Phase 3: Quick Start Guide

## TL;DR

**The KG library is complete (23 files, 4,598 lines). Phase 3 extracts utility patterns from Phase 2 scripts.**

- **What to build:** 5 utility modules in `mellea_contribs/kg/utils/`
- **What NOT to build:** Anything in core library (it's done!)
- **Time estimate:** 0.5-1 day (mostly copy-paste)
- **Lines of code:** ~650 (NOT 800-1000)

---

## Files to Create

### 1. mellea_contribs/kg/utils/__init__.py (50 lines)
**Purpose:** Export all utilities
```python
from .data_utils import load_jsonl, save_jsonl, batch_iterator, ...
from .session_manager import create_session, create_backend, ...
from .progress import setup_logging, log_progress, output_json, ...
from .eval_utils import exact_match, mean_reciprocal_rank, ...

__all__ = [
    "load_jsonl", "save_jsonl", "batch_iterator",
    "create_session", "create_backend",
    "setup_logging", "log_progress", "output_json",
    "exact_match", "mean_reciprocal_rank",
]
```

### 2. mellea_contribs/kg/utils/data_utils.py (200 lines)
**Purpose:** JSONL I/O and batch processing

**Copy these patterns from Phase 2:**
- JSONL reading/writing: create_demo_dataset.py, run_kg_preprocess.py
- Batch iteration: run_kg_embed.py, run_qa.py

**Functions to implement:**
```python
def load_jsonl(path: Path) -> Iterator[dict]
def save_jsonl(data: List[dict], path: Path) -> None
def append_jsonl(item: dict, path: Path) -> None
def batch_iterator(items: List, batch_size: int) -> Iterator[List]
def truncate_jsonl(input_path, output_path, max_lines)
def shuffle_jsonl(input_path, output_path)
def compress_jsonl(input_path, output_path)
def decompress_jsonl(input_path, output_path)
```

### 3. mellea_contribs/kg/utils/session_manager.py (100 lines)
**Purpose:** Mellea session and backend creation

**Copy these patterns from Phase 2:**
- build_backend(): All run_*.py files (line ~47)
- Session creation: All run_*.py files (line ~203)

**Functions to implement:**
```python
def create_session(backend_name="litellm", model_id="gpt-4o-mini", ...) -> MelleaSession
def create_backend(backend_type="mock", neo4j_uri=None, ...) -> GraphBackend

class MelleaResourceManager:
    async def __aenter__()
    async def __aexit__()
```

### 4. mellea_contribs/kg/utils/progress.py (150 lines)
**Purpose:** Logging, progress tracking, structured output

**Copy these patterns from Phase 2:**
- stderr output: Any run_*.py file
- Stats printing: run_eval.py (lines 183-196)

**Functions to implement:**
```python
def setup_logging(log_level="INFO", log_file=None) -> None
def log_progress(msg: str, level="INFO") -> None
def output_json(obj: BaseModel) -> None
def print_stats(stats: Union[QAStats, UpdateStats, EmbeddingStats]) -> None

class ProgressTracker:
    def __init__(self, total: int, desc: str)
    def update(self, n: int)
    def close()
```

### 5. mellea_contribs/kg/utils/eval_utils.py (150 lines)
**Purpose:** Evaluation metrics

**Copy these functions from Phase 2:**
- compute_mrr(): run_eval.py (lines 44-63)
- compute_exact_match(): run_eval.py (lines 27-29)
- Stats aggregation: run_eval.py, run_qa.py

**Functions to implement:**
```python
def exact_match(predicted: str, expected: str) -> bool
def fuzzy_match(predicted: str, expected: str, threshold=0.8) -> bool
def mean_reciprocal_rank(results: List[dict]) -> float
def precision(predicted, expected) -> float
def recall(predicted, expected) -> float
def f1_score(precision, recall) -> float
def aggregate_qa_results(qa_results: List[QAResult]) -> QAStats
def aggregate_update_results(update_results: List[UpdateResult]) -> UpdateStats
```

---

## Implementation Order

1. **data_utils.py** (Most straightforward - copy JSONL patterns)
2. **session_manager.py** (Copy backend factory + session creation)
3. **eval_utils.py** (Copy metrics from run_eval.py)
4. **progress.py** (Copy logging patterns)
5. **__init__.py** (Write imports)

---

## What Phase 3 is NOT

- ❌ NOT reimplementing extraction (already in components/generative.py)
- ❌ NOT creating query generation (already in components/llm_guided.py)
- ❌ NOT building new backends (already in graph_dbs/)
- ❌ NOT creating config models (already in qa_models.py, updater_models.py)
- ❌ NOT modifying existing files (keep Phase 1-2 untouched)

---

## Reference: Where to Copy Patterns From

### data_utils.py patterns:
- **JSONL write:** docs/examples/kgrag/scripts/create_demo_dataset.py (lines 40-60)
- **JSONL read:** docs/examples/kgrag/scripts/run_kg_preprocess.py (lines 78-100)
- **Batch iterate:** docs/examples/kgrag/scripts/run_qa.py (lines 71-90)

### session_manager.py patterns:
- **build_backend:** docs/examples/kgrag/scripts/run_kg_preprocess.py (lines 47-60)
- **Session creation:** docs/examples/kgrag/scripts/run_qa.py (line 240)

### progress.py patterns:
- **stderr output:** Any run_*.py file (search for `file=sys.stderr`)
- **Stats printing:** docs/examples/kgrag/scripts/run_eval.py (lines 183-196)

### eval_utils.py patterns:
- **All metrics:** docs/examples/kgrag/scripts/run_eval.py (entire file)

---

## Testing Phase 3

After implementation, verify by:

```python
# Test imports
from mellea_contribs.kg.utils import (
    load_jsonl, save_jsonl, batch_iterator,
    create_session, create_backend,
    setup_logging, log_progress, output_json,
    exact_match, mean_reciprocal_rank
)

# Test data_utils
items = list(load_jsonl("test.jsonl"))

# Test session_manager
backend = create_backend(backend_type="mock")
session = create_session(model_id="gpt-4o-mini")

# Test progress
setup_logging()
log_progress("Test message")

# Test eval
score = exact_match("answer", "answer")
mrr = mean_reciprocal_rank([...])
```

---

## Optional: Update Phase 2 Scripts (Nice-to-Have)

After Phase 3 utilities are ready, optionally refactor Phase 2 scripts to use them:

```python
# Before
from pathlib import Path
import json

with open(path) as f:
    for line in f:
        item = json.loads(line)

# After
from mellea_contribs.kg.utils import load_jsonl

for item in load_jsonl(path):
    ...
```

This is optional - Phase 3 is complete without this refactoring.

---

## Summary

**Phase 3 = Extract + Organize patterns from Phase 2 into reusable utility library**

Time estimate: 0.5-1 day
Code needed: ~650 lines
Complexity: Low (mostly copy-paste)

Start with data_utils.py and work through to __init__.py.

Good luck! 🎉
