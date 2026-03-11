# Phase 3: Expectations vs. Reality

## What We Thought Phase 3 Would Be

```
Original Plan (based on PR#3 needs):
├── Data utilities (8 modules)
│   ├── movie_dataset.py - Custom dataset loader
│   ├── data.py - Custom JSONL utilities
│   ├── utils_mellea.py - Custom session creation
│   └── utils.py - Custom metrics, retry logic
├── Logging infrastructure (3 modules)
│   ├── logger.py - Custom logger classes
│   ├── prompt_list.py - Custom prompt templates
│   └── Configuration templates
└── Total effort: 1-1.5 days (800-1000 lines of custom code)
```

**Assumption:** These functions don't exist yet and must be built from scratch.

---

## What Phase 3 Actually Is (Reality Check)

### The KG Library Discovery: 23 Files, 4,598 Lines Already Built

```
LAYER 1: Orchestration (951 lines)
├── kgrag.py - Full QA orchestrator ✓
├── preprocessor.py - Document processing ✓
├── embedder.py - Entity embeddings ✓
└── rep.py - Entity representation ✓

LAYER 2: Configuration (793 lines)
├── qa_models.py - QA config + QAResult ✓
├── updater_models.py - Update config + UpdateResult ✓
├── embed_models.py - Embedding config ✓
└── requirements_models.py - Validation factories ✓

LAYER 3: Generative Functions (639 lines)
├── components/generative.py - Extract, align, decide, validate ✓
└── components/llm_guided.py - Query generation & repair ✓

LAYER 4: Backends (532 lines)
├── graph_dbs/base.py - Abstract interface ✓
├── graph_dbs/mock.py - Testing backend ✓
└── graph_dbs/neo4j.py - Production backend ✓

PLUS: Graph data structures, components, sampling, validation
```

**Key Finding:** The infrastructure is COMPLETE. Phase 3 needs utilities, NOT core components.

---

## Phase 3 Redefined: Utility Extraction & Organization

### What Phase 3 Actually Does

**Before (Phase 2 - Scattered Patterns):**
```python
# run_kg_preprocess.py has:
def load_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)

# run_qa.py has similar code
# run_kg_update.py has similar code
# run_eval.py has similar code
```

**After (Phase 3 - Centralized Utilities):**
```python
# mellea_contribs/kg/utils/data_utils.py
def load_jsonl(path):
    """Reusable JSONL loader used everywhere"""
    with open(path) as f:
        for line in f:
            yield json.loads(line)

# All Phase 2 scripts now import from utils
from mellea_contribs.kg.utils import load_jsonl
```

### Phase 3 Reality: ~600 Lines of Copy-Paste + Organization

| Module | Source | Lines | Effort | Complexity |
|--------|--------|-------|--------|-----------|
| **data_utils.py** | create_*.py, run_*.py | 200 | Copy JSONL patterns | Low |
| **session_manager.py** | build_backend() in run_*.py | 100 | Copy backend factory | Low |
| **progress.py** | stderr/stdout in run_*.py | 150 | Copy logging patterns | Low |
| **eval_utils.py** | run_eval.py | 150 | Copy metrics functions | Low |
| **__init__.py** | Manual | 50 | Write imports | Low |
| **TOTAL** | - | **650** | **0.5-1 day** | **All copy-paste** |

---

## What Phase 3 is NOT

### ❌ NOT Building Missing Infrastructure

| What We Thought | What's Actually True |
|-----------------|---------------------|
| Need to build entity extraction | ✓ Already in components/generative.py (549 LOC) |
| Need to build query generation | ✓ Already in components/llm_guided.py (90 LOC) |
| Need to build database backends | ✓ Already in graph_dbs/ (532 LOC total) |
| Need to create config models | ✓ Already in qa_models.py + updater_models.py (659 LOC) |
| Need to implement logging from scratch | ✓ Already in all Layer 1 apps (standard logging) |
| Need custom graph data structures | ✓ Already in base.py (103 LOC) |

### ❌ NOT Reimplementing Extraction/Query Logic

**What Phase 3 WILL NOT do:**
- ❌ Create new LLM extraction functions
- ❌ Build new query generation
- ❌ Create new database abstraction
- ❌ Build new configuration models
- ❌ Reimplement entity/relation models

**Why:** All of this already works. Phase 3 should not touch it.

---

## Phase 3 is REALLY About This

### Before Phase 3 (Scattered Utility Code)
```
docs/examples/kgrag/scripts/
├── create_demo_dataset.py (contains JSONL writing logic)
├── run_kg_preprocess.py (contains JSONL reading logic)
├── run_kg_embed.py (contains session creation logic)
├── run_qa.py (contains progress tracking logic)
├── run_kg_update.py (contains backend creation logic)
└── run_eval.py (contains metrics computation logic)
```

**Problem:** Each script reimplements the same utilities differently.

### After Phase 3 (Centralized Utilities)
```
mellea_contribs/kg/utils/
├── data_utils.py (JSONL reading/writing - used by all scripts)
├── session_manager.py (Session creation - used by all scripts)
├── progress.py (Progress tracking - used by all scripts)
├── eval_utils.py (Metrics - used by evaluation scripts)
└── __init__.py (Unified exports)

docs/examples/kgrag/scripts/
├── create_demo_dataset.py (import load_jsonl, save_jsonl)
├── run_kg_preprocess.py (import load_jsonl, create_session)
├── run_kg_embed.py (import create_session, ProgressTracker)
├── run_qa.py (import load_jsonl, ProgressTracker)
├── run_kg_update.py (import create_backend, log_progress)
└── run_eval.py (import load_jsonl, compute_metrics)
```

**Benefit:** DRY principle, consistent behavior, easier maintenance.

---

## The Real Scope of Phase 3

### It's a Library Refactoring, Not Feature Development

```
Phase 1: Build core KG infrastructure       ✓ DONE (2700 LOC)
Phase 2: Build CLI scripts that use it      ✓ DONE (1557 LOC)
Phase 3: Extract utilities from scripts     → NEXT (650 LOC)
         and organize into reusable lib

Phase 4: Configuration & templates          → AFTER (100-200 LOC)
```

**Phase 3 is about making Phase 2 code DRY (Don't Repeat Yourself).**

---

## Effort Estimation: Corrected

| Phase | Expected | Actual | Changed | Status |
|-------|----------|--------|---------|--------|
| 1 | 2700 LOC | 2700 LOC | ✓ On track | DONE |
| 2 | 1500-2000 LOC | 1557 LOC | ✓ On track | DONE |
| 3 | 800-1000 LOC | **600-800 LOC** | **-200 LOC** | READY |
| 4 | 100-200 LOC | 100-200 LOC | ✓ On track | Simple |
| **Total** | 5100-5900 LOC | **5157-5257 LOC** | **-400 LOC** | 1 Day |

**Why it's smaller:** Phase 3 is mostly copy-paste from Phase 2, not new development.

---

## Action Items for Phase 3

### Step 1: Create Directory Structure (5 mins)
```bash
mkdir -p mellea_contribs/kg/utils
touch mellea_contribs/kg/utils/__init__.py
```

### Step 2: Create data_utils.py (30 mins)
Copy these patterns from Phase 2:
- JSONL reading: from create_*.py (lines 40-60)
- JSONL writing: from create_*.py (lines 50-70)
- Batch iteration: from run_*.py (lines 80-100)

### Step 3: Create session_manager.py (20 mins)
Copy these patterns from Phase 2:
- build_backend(): from any run_*.py (lines 45-60)
- Session creation: from any run_*.py (line 203)

### Step 4: Create progress.py (30 mins)
Copy these patterns from Phase 2:
- Logging setup: from any run_*.py
- stderr output: print(..., file=sys.stderr)
- stdout JSON: json.dumps(obj.model_dump())

### Step 5: Create eval_utils.py (30 mins)
Copy these functions from Phase 2:
- compute_mrr(): from run_eval.py (lines 44-63)
- compute_exact_match(): from run_eval.py (lines 27-29)
- Other metric functions

### Step 6: Create __init__.py (10 mins)
Export all utilities

### Step 7: Update Imports in Phase 2 Scripts (20 mins, OPTIONAL)
Change run_*.py scripts to use new utilities (optional enhancement)

**Total Time: 2-3 hours of work**

---

## Key Takeaway

**Phase 3 is NOT about implementing missing functionality.**

The KG library is complete and production-ready. Phase 3 is about taking the working patterns demonstrated in Phase 2 scripts and organizing them into reusable, maintainable utility libraries.

**This is good engineering practice: extract, generalize, reuse.**

---

## Files to Reference During Phase 3 Implementation

### For data_utils.py:
- docs/examples/kgrag/scripts/create_demo_dataset.py (lines 40-60)
- docs/examples/kgrag/scripts/run_kg_preprocess.py (lines 78-100)
- docs/examples/kgrag/scripts/run_qa.py (lines 71-90)

### For session_manager.py:
- docs/examples/kgrag/scripts/run_kg_preprocess.py (lines 47-60, 203)
- docs/examples/kgrag/scripts/run_qa.py (lines 239-240)

### For progress.py:
- Any run_*.py file (stderr/stdout patterns)
- docs/examples/kgrag/scripts/run_eval.py (lines 183-196)

### For eval_utils.py:
- docs/examples/kgrag/scripts/run_eval.py (entire file)
- docs/examples/kgrag/scripts/run_qa.py (lines 118-120, 155-162)
