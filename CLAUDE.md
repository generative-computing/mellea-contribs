# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mellea-contribs** is an incubation repository for contributions to the Mellea ecosystem. It provides a library for incubating generative programming tools and utilities that integrate with the Mellea LLM agent framework.

- **Tech Stack**: Python 3.10+, PDM build system, Mellea framework with LiteLLM
- **Key Dependencies**: rapidfuzz (fuzzy matching), eyecite (legal citations), playwright (web scraping), markdown
- **License**: Apache License 2.0

## Common Development Commands

### Setup and Installation

```bash
# Install with development dependencies
pdm install --group dev

# Or using uv (faster):
uv pip install -e . --group dev
```

### Code Quality

```bash
# Format code (ruff is used for both formatting and linting)
ruff format .
ruff check . --fix

# Type checking
mypy mellea_contribs/

# Run all linters individually
isort mellea_contribs/ test/
pylint mellea_contribs/ test/
ruff check mellea_contribs/ test/

# Run pre-commit hooks
pre-commit run --all-files
```

### Testing

```bash
# Run all tests
pytest

# Run a specific test file
pytest test/test_citation_exists.py

# Run a specific test function
pytest test/test_citation_exists.py::test_function_name

# Run tests with verbose output
pytest -v

# Run only tests marked as qualitative (LLM-dependent)
pytest -m qualitative

# Run tests excluding qualitative tests (useful in CI)
pytest -m "not qualitative"

# Run tests with asyncio verbose output (asyncio_mode is auto-configured)
pytest -v --asyncio-mode=auto
```

### Documentation

```bash
# Build Sphinx documentation
sphinx-build -b html docs/ docs/_build/
```

## Architecture Overview

### Module Organization

**`mellea_contribs/tools/`** - Reusable LLM-based selection and ranking algorithms:
- `top_k.py`: Generic Top-K selection engine using LLM judgment. Selects top K from N items with rejection sampling and caching.
- `double_round_robin.py`: Pairwise comparison scoring engine. Performs all-pair comparisons and returns items ranked by accumulated scores.

**`mellea_contribs/reqlib/`** - Domain-specific validators extending Mellea's validation framework:
- `citation_exists.py`: Validates legal case citations via case.law metadata + fuzzy matching
- `is_appellate_case.py`: Classifies cases as appellate by court abbreviation patterns
- `import_repair.py`: Fixes Python import errors in LLM-generated code via AST analysis
- `import_resolution.py`: Parses and resolves module not found / import errors with confidence-scored suggestions
- `grounding_context_formatter.py`: Structures multi-field context for LLM prompts (auto-skips empty fields)
- `common_aliases.py`: Module name mappings and relocations for import resolution
- `statute_data.py`: Legal statute data handling

**`mellea_contribs/kg/`** - Knowledge Graph database abstraction:
- `base.py`: Core data structures (GraphNode, GraphEdge, GraphPath)
- `graph_dbs/base.py`: GraphBackend abstract interface
- `graph_dbs/neo4j.py`: Production Neo4j implementation (requires [kg] dependencies)
- `graph_dbs/mock.py`: In-memory mock backend for testing without infrastructure
- `components/`: Query, result, traversal components (minimal Layer 4 implementations)

**Installation**: `pip install mellea-contribs[kg]` for Neo4j support

### Design Patterns

**Mellea Integration Pattern**: All validators are Requirement classes with a `validation_fn(output) → ValidationResult`. This enables iterative LLM refinement via the Instruct-Validate-Repair loop.

**Caching Strategy**: Tools use decorator-based caching keyed on item hash + context + prompts to avoid redundant LLM calls.

**Model Interaction**: Uses `mellea.instruct()` with grounding context, system prompts for output formatting (JSON arrays, single tokens), and rejection sampling (loop_budget=2) for reliability.

**Data Validation Layers**:
- Legal citations: Direct lookup → fuzzy match → LLM judgment
- Python imports: Static AST analysis + dynamic error parsing
- Court classification: Pattern matching + database lookup

### Test Infrastructure

- Large test databases: `test/data/citation_exists_database.json` (~2.8MB)
- Qualitative test markers for LLM-dependent tests (auto-xfail in CI when `MELLEA_SKIP_QUALITATIVE` env var is set)
- Neo4j integration tests: Marked with `@pytest.mark.neo4j`, skipped unless NEO4J_URI is set
- Separate database for CI efficiency
- Shared fixtures via `test/conftest.py`
- Async-first testing: `asyncio_mode = "auto"` (no explicit marking needed)

## Code Quality Standards

- **Docstrings**: Google-style convention (enforced via ruff rule D)
- **Complexity**: Maximum cyclomatic complexity of 20 (ruff C901)
- **Type Hints**: Full type annotation with mypy checking enabled
- **Imports**: isort enforced for consistent ordering with `combine-as-imports`
- **Pre-commit hooks**: Automatic formatting and validation on commit

## Release and Versioning

- **Semantic Versioning**: Angular commit parser (feat → minor, fix → patch)
- **Automated Releases**: python-semantic-release on main branch
- **CI/CD**: GitHub Actions workflows (ci.yml, cd.yml, pypi.yml)
- **PyPI Publishing**: Automated on semantic version tags
