# Phase 3 Utility Tests

## Overview

Comprehensive test suite for the Phase 3 utility modules in `mellea_contribs/kg/utils/`. This test suite ensures that all utility functions work correctly and integrates with the rest of the KG system.

## Test Files

### 1. test_data_utils.py (27 tests)
Tests for JSONL I/O and batch processing utilities.

**Test Classes:**
- `TestLoadAndSaveJsonl` - Reading/writing JSONL files
- `TestAppendJsonl` - Appending items to JSONL files
- `TestBatchIterator` - Batch iteration functionality
- `TestTruncateJsonl` - Truncating JSONL files
- `TestShuffleJsonl` - Shuffling JSONL data
- `TestValidateJsonlSchema` - Schema validation
- `TestIntegration` - Integration workflows

**Coverage:**
- ✓ Empty files, single items, multiple items
- ✓ Error handling (nonexistent files, invalid JSON)
- ✓ Directory creation
- ✓ Batch sizes (exact, uneven, edge cases)
- ✓ Truncation at various limits
- ✓ Schema validation with missing fields
- ✓ Integration workflows

### 2. test_eval_utils.py (26 tests)
Tests for evaluation metrics and result aggregation.

**Test Classes:**
- `TestExactMatch` - Exact string matching
- `TestFuzzyMatch` - Fuzzy string matching
- `TestMeanReciprocalRank` - MRR computation
- `TestPrecisionRecall` - Precision and recall metrics
- `TestF1Score` - F1 score computation
- `TestAggregateQaResults` - QA result aggregation
- `TestAggregateUpdateResults` - Update result aggregation
- `TestIntegration` - Integration workflows

**Coverage:**
- ✓ Case-insensitive matching
- ✓ Whitespace handling
- ✓ Threshold sensitivity
- ✓ Empty results handling
- ✓ Perfect/partial/no matches
- ✓ Confidence-based ranking
- ✓ Classification metrics (precision, recall, F1)
- ✓ Result aggregation with errors

### 3. test_progress.py (23 tests)
Tests for logging, progress tracking, and structured output.

**Test Classes:**
- `TestSetupLogging` - Logging configuration
- `TestLogProgress` - Progress message logging
- `TestOutputJson` - JSON output
- `TestPrintStats` - Statistics formatting
- `TestProgressTracker` - Progress tracking class
- `TestIntegration` - Integration workflows

**Coverage:**
- ✓ Logging levels (DEBUG, INFO, WARNING, ERROR)
- ✓ File logging
- ✓ JSON serialization of Pydantic models
- ✓ Statistics pretty-printing
- ✓ Indentation and formatting
- ✓ ProgressTracker initialization and updates
- ✓ Integration workflows

### 4. test_session_manager.py (19 tests)
Tests for session and backend creation.

**Test Classes:**
- `TestCreateBackend` - Backend factory function
- `TestCreateSession` - Session factory function
- `TestMelleaResourceManager` - Async resource manager
- `TestIntegration` - Integration workflows

**Coverage:**
- ✓ Mock backend creation
- ✓ Default parameters
- ✓ Invalid backend types
- ✓ Neo4j backend (when available)
- ✓ Custom parameters
- ✓ Async context manager cleanup
- ✓ Integration workflows

## Running Tests

### Run all Phase 3 tests:
```bash
pytest test/kg/utils/ -v
```

### Run specific test file:
```bash
pytest test/kg/utils/test_data_utils.py -v
```

### Run specific test class:
```bash
pytest test/kg/utils/test_data_utils.py::TestLoadAndSaveJsonl -v
```

### Run specific test:
```bash
pytest test/kg/utils/test_data_utils.py::TestLoadAndSaveJsonl::test_save_and_load_single_item -v
```

## Test Statistics

- **Total Tests:** 95
- **Passed:** 95 ✓
- **Failed:** 0
- **Coverage:** Comprehensive coverage of all Phase 3 utilities

## Test Strategy

### Data Utilities (data_utils.py)
- **Boundary testing:** Empty files, single items, multiple items
- **Error handling:** Invalid JSON, missing files
- **Integration:** Workflows combining multiple utilities
- **Edge cases:** Batch sizes, truncation limits, shuffling order preservation

### Evaluation Utilities (eval_utils.py)
- **Matching:** Exact, fuzzy, case-insensitive, whitespace handling
- **Metrics:** Precision, recall, F1, MRR computation
- **Aggregation:** Batch result aggregation with errors
- **Edge cases:** Empty results, perfect/no matches, confidence scoring

### Progress Utilities (progress.py)
- **Logging:** Multiple levels, file output
- **Output:** JSON serialization, pretty-printing
- **Tracking:** Progress bar updates
- **Formatting:** Indentation, stats display

### Session Manager (session_manager.py)
- **Backend creation:** Mock, Neo4j (conditional)
- **Session creation:** Default/custom parameters
- **Resource management:** Async context managers
- **Integration:** Workflow testing

## Dependencies

- pytest
- pytest-asyncio
- mellea_contribs.kg utilities
- Pydantic models (QAStats, UpdateStats, etc.)

## Notes

- All tests use temporary directories for file operations
- Async tests properly use pytest-asyncio
- Tests gracefully handle optional dependencies (Neo4j)
- Integration tests verify workflows combining multiple utilities
