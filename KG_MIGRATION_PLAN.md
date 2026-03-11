# KG Library Migration Plan: mellea PR#2 → mellea-contribs

## Overview

Migrate the Knowledge Graph (KG) library from https://github.com/ydzhu98/mellea/pull/2 into mellea-contribs, preserving its Layer 4 architecture while adapting to mellea-contribs patterns.

**Source**: mellea/contribs/kg/ (PR#2)
**Destination**: mellea_contribs/kg/
**Components**: 30 tests, core data structures, Neo4j + Mock backends, design docs

## Critical Files to Modify

1. `/home/yzhu/mellea-contribs/pyproject.toml` - Add [kg] dependency group, neo4j pytest marker
2. `/home/yzhu/mellea-contribs/test/conftest.py` - Add neo4j test skip logic
3. All source files in `mellea_contribs/kg/` - Import path updates (mellea.contribs.kg → mellea_contribs.kg)
4. All test files in `test/kg/` - Import updates, pytest markers
5. Documentation files in `docs/` - Convert design doc to .mdx format

## Implementation Steps

### Phase 1: Project Configuration (pyproject.toml)

**File**: `/home/yzhu/mellea-contribs/pyproject.toml`

Add after line 55 (in dependency-groups section):
```toml
kg = [
    "neo4j>=5.0.0",
]
```

Update pytest markers (after line 127):
```toml
[tool.pytest.ini_options]
markers = [
    "qualitative: Marks the test as needing an exact output from an LLM; set by an ENV variable for CICD. All tests marked with this will xfail in CI/CD",
    "neo4j: Marks tests requiring a running Neo4j instance (skipped in CI unless NEO4J_URI is set)",
]
```

### Phase 2: Directory Structure Creation

Create these directories:
```
mellea_contribs/kg/
├── components/
├── graph_dbs/
├── requirements/
└── sampling/

test/kg/

docs/ (already exists)
```

### Phase 3: Source Files Migration

**Migration order (for dependency resolution):**

1. **Base data structures** (no dependencies):
   - Source: `mellea/contribs/kg/base.py`
   - Dest: `mellea_contribs/kg/base.py`
   - Changes: Add module docstring, verify type hints

2. **Backend interface**:
   - Source: `mellea/contribs/kg/graph_dbs/base.py`
   - Dest: `mellea_contribs/kg/graph_dbs/base.py`
   - Changes: Update import `mellea.contribs.kg.base` → `mellea_contribs.kg.base`

3. **Mock backend**:
   - Source: `mellea/contribs/kg/graph_dbs/mock.py`
   - Dest: `mellea_contribs/kg/graph_dbs/mock.py`
   - Changes: Update imports to `mellea_contribs`

4. **Neo4j backend** (requires special handling):
   - Source: `mellea/contribs/kg/graph_dbs/neo4j.py`
   - Dest: `mellea_contribs/kg/graph_dbs/neo4j.py`
   - Changes:
     - Update imports to `mellea_contribs`
     - Add optional import handling:
       ```python
       try:
           from neo4j import GraphDatabase
           NEO4J_AVAILABLE = True
       except ImportError:
           NEO4J_AVAILABLE = False
           GraphDatabase = None

       class Neo4jBackend(GraphBackend):
           def __init__(self, uri, user, password):
               if not NEO4J_AVAILABLE:
                   raise ImportError(
                       "Neo4j support requires additional dependencies. "
                       "Install with: pip install mellea-contribs[kg]"
                   )
       ```

5. **Component stubs**:
   - Source: `mellea/contribs/kg/components/*.py`
   - Dest: `mellea_contribs/kg/components/*.py`
   - Changes: Update imports to `mellea_contribs`

### Phase 4: Create __init__.py Files

**File**: `mellea_contribs/kg/__init__.py`
```python
"""Knowledge Graph library for mellea-contribs.

Optional Dependencies:
    Neo4j support requires: pip install mellea-contribs[kg]
"""

from mellea_contribs.kg.base import GraphEdge, GraphNode, GraphPath
from mellea_contribs.kg.graph_dbs.base import GraphBackend
from mellea_contribs.kg.graph_dbs.mock import MockGraphBackend

try:
    from mellea_contribs.kg.graph_dbs.neo4j import Neo4jBackend
except ImportError as e:
    def Neo4jBackend(*args, **kwargs):
        raise ImportError(
            "Neo4j support requires additional dependencies. "
            "Install with: pip install mellea-contribs[kg]"
        ) from e

__all__ = [
    "GraphNode", "GraphEdge", "GraphPath",
    "GraphBackend", "Neo4jBackend", "MockGraphBackend",
]
```

**File**: `mellea_contribs/kg/graph_dbs/__init__.py`
```python
"""Graph database backend implementations."""

from mellea_contribs.kg.graph_dbs.base import GraphBackend
from mellea_contribs.kg.graph_dbs.mock import MockGraphBackend

try:
    from mellea_contribs.kg.graph_dbs.neo4j import Neo4jBackend
except ImportError:
    Neo4jBackend = None

__all__ = ["GraphBackend", "MockGraphBackend", "Neo4jBackend"]
```

**File**: `mellea_contribs/kg/components/__init__.py`
```python
"""Query components for graph database operations."""

from mellea_contribs.kg.components.query import GraphQuery
from mellea_contribs.kg.components.result import GraphResult
from mellea_contribs.kg.components.traversal import GraphTraversal

__all__ = ["GraphQuery", "GraphResult", "GraphTraversal"]
```

**Empty files**:
- `mellea_contribs/kg/requirements/__init__.py`
- `mellea_contribs/kg/sampling/__init__.py`
- `test/kg/__init__.py`

### Phase 5: Test Files Migration

**Order**:

1. **test_base.py** (9 tests, no external dependencies):
   - Source: `test/contribs/kg/test_base.py`
   - Dest: `test/kg/test_base.py`
   - Changes: Update imports to `mellea_contribs.kg`

2. **test_mock_backend.py** (7 tests, no external dependencies):
   - Source: `test/contribs/kg/test_mock_backend.py`
   - Dest: `test/kg/test_mock_backend.py`
   - Changes: Update imports to `mellea_contribs.kg`

3. **test_neo4j_backend.py** (14 tests, requires Neo4j):
   - Source: `test/contribs/kg/test_neo4j_backend.py`
   - Dest: `test/kg/test_neo4j_backend.py`
   - Changes:
     - Update imports to `mellea_contribs.kg`
     - Add at top of file:
       ```python
       import os
       import pytest

       # Skip all tests if Neo4j not installed
       pytest.importorskip("neo4j", reason="Neo4j not installed. Install with: pip install mellea-contribs[kg]")

       # Mark all tests in this module
       pytestmark = pytest.mark.neo4j

       @pytest.fixture(scope="module")
       def neo4j_backend():
           """Fixture providing Neo4j backend connection."""
           uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
           user = os.getenv("NEO4J_USER", "neo4j")
           password = os.getenv("NEO4J_PASSWORD", "password")

           try:
               backend = Neo4jBackend(uri, user, password)
               yield backend
               backend.close()
           except Exception as e:
               pytest.skip(f"Cannot connect to Neo4j: {e}")
       ```

### Phase 6: Update conftest.py

**File**: `/home/yzhu/mellea-contribs/test/conftest.py`

Add after line 24 in `pytest_runtest_setup` function:
```python
def pytest_runtest_setup(item):
    """Skip qualitative tests when running in CI environment."""
    if not item.get_closest_marker("qualitative"):
        # Check for neo4j marker
        if item.get_closest_marker("neo4j"):
            if not os.environ.get("NEO4J_URI"):
                pytest.skip(
                    reason="Skipping neo4j test: NEO4J_URI environment variable not set. "
                    "Set NEO4J_URI to enable Neo4j integration tests."
                )
        return

    gh_run = int(os.environ.get("CICD", 0))
    if gh_run == 1:
        pytest.skip(
            reason="Skipping qualitative test: got env variable CICD == 1. Used only in gh workflows."
        )
```

### Phase 7: Documentation Migration

Create these files in `/home/yzhu/mellea-contribs/docs/`:

1. **kg_overview.mdx** - Architecture, quick start, when to use
2. **kg_backends.mdx** - Backend implementations, Neo4j config, Mock usage
3. **kg_quick_start.mdx** - Installation, basic examples
4. **kg_api_reference.mdx** - Complete API docs

Source content from: `docs/design/graph_query_library.md` (1,597 lines)

**File**: `mellea_contribs/kg/README.md`
```markdown
# Knowledge Graph (KG) Library

Backend-agnostic graph database abstraction for Mellea applications.

## Installation

```bash
# Basic installation (includes MockGraphBackend)
pip install mellea-contribs

# With Neo4j support
pip install mellea-contribs[kg]
```

## Quick Example

```python
from mellea_contribs.kg import Neo4jBackend

backend = Neo4jBackend(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

node1 = backend.create_node({"name": "Alice"})
node2 = backend.create_node({"name": "Bob"})
backend.create_edge(node1, node2, "KNOWS")
```

## Documentation

See [full documentation](../../docs/kg_overview.mdx) for complete guides.

## Status

- ✓ Layer 4: Backend abstraction (complete)
- ⧗ Layer 5: KGRag integration (future work)
```

### Phase 8: Update Project Documentation

**File**: `/home/yzhu/mellea-contribs/CLAUDE.md`

Add after line 91 (in Module Organization):
```markdown
**`mellea_contribs/kg/`** - Knowledge Graph database abstraction:
- `base.py`: Core data structures (GraphNode, GraphEdge, GraphPath)
- `graph_dbs/base.py`: GraphBackend abstract interface
- `graph_dbs/neo4j.py`: Production Neo4j implementation (requires [kg] dependencies)
- `graph_dbs/mock.py`: In-memory mock for testing
- `components/`: Query, result, traversal components (stubs for Layer 5)

**Installation**: `pip install mellea-contribs[kg]` for Neo4j support
```

Add after line 113 (in Test Infrastructure):
```markdown
- Neo4j integration tests: Marked with `@pytest.mark.neo4j`, skipped unless NEO4J_URI is set
```

**File**: `/home/yzhu/mellea-contribs/README.md`

Add after line 8:
```markdown
## Installation

```bash
# Basic installation
pip install mellea-contribs

# With optional dependencies
pip install mellea-contribs[kg]        # Knowledge Graph / Neo4j support
pip install mellea-contribs[dev]       # Development tools
pip install mellea-contribs[docs]      # Documentation building
```

## Modules

### Knowledge Graph (KG)
Graph database abstraction for KG-RAG applications. Provides backend-agnostic interface with Neo4j implementation and mock backend for testing.

- [Documentation](./docs/kg_overview.mdx)
- Install: `pip install mellea-contribs[kg]`
```

## Verification Steps

### Step 1: Import Verification (without [kg] dependencies)
```bash
cd /home/yzhu/mellea-contribs

# Verify base imports work
python -c "from mellea_contribs.kg import GraphNode, GraphEdge, GraphPath, MockGraphBackend; print('✓ Base imports OK')"

# Verify Neo4j import fails gracefully
python -c "
try:
    from mellea_contribs.kg import Neo4jBackend
    Neo4jBackend('test', 'test', 'test')
    print('✗ Should have raised ImportError')
except ImportError as e:
    print('✓ Expected import error:', str(e))
"
```

Expected: Both tests pass, second shows helpful error message.

### Step 2: Run Base Tests
```bash
pytest test/kg/test_base.py test/kg/test_mock_backend.py -v
```

Expected: 16 tests pass (9 base + 7 mock)

### Step 3: Install [kg] and Test
```bash
pip install -e ".[kg]"

# Verify Neo4j import now works
python -c "from mellea_contribs.kg import Neo4jBackend; print('✓ Neo4j import OK')"

# Run all tests (without Neo4j connection)
pytest test/kg/ -v
```

Expected: 16 passed, 14 skipped (Neo4j tests require NEO4J_URI)

### Step 4: Code Quality Checks
```bash
# Format and lint
ruff format mellea_contribs/kg/ test/kg/
ruff check mellea_contribs/kg/ test/kg/ --fix

# Type checking
mypy mellea_contribs/kg/

# Import ordering
isort mellea_contribs/kg/ test/kg/

# Pre-commit hooks
pre-commit run --files mellea_contribs/kg/**/*.py test/kg/**/*.py
```

Expected: All checks pass

### Step 5: Optional Neo4j Integration Test
```bash
# Start Neo4j (Docker example)
docker run -d --name neo4j-test -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/testpassword neo4j:5.0

# Run full test suite
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=testpassword

pytest test/kg/ -v

# Cleanup
docker stop neo4j-test && docker rm neo4j-test
```

Expected: All 30 tests pass

## Git Commit Strategy

**Commit 1**: Project configuration
```
feat(kg): add KG library dependency configuration

- Add [kg] dependency group with neo4j>=5.0.0
- Add @pytest.mark.neo4j marker
- Update conftest.py with Neo4j skip logic
```

**Commit 2**: Core implementation
```
feat(kg): add Knowledge Graph library (Layer 4)

- Add GraphNode, GraphEdge, GraphPath data structures
- Add GraphBackend abstract interface
- Add Neo4jBackend with optional dependency handling
- Add MockGraphBackend for testing
- Add component stubs (query, result, traversal)
- Migrate from https://github.com/ydzhu98/mellea/pull/2
```

**Commit 3**: Tests
```
test(kg): add comprehensive KG library test suite

- Add 9 base data structure tests
- Add 7 mock backend tests
- Add 14 Neo4j integration tests with proper markers
- All tests pass (16 without Neo4j, 30 with Neo4j)
```

**Commit 4**: Documentation
```
docs(kg): add KG library documentation

- Add kg_overview.mdx with architecture
- Add kg_backends.mdx with implementation details
- Add kg_quick_start.mdx with examples
- Add kg_api_reference.mdx with API docs
- Add README.md in kg/ directory
- Update CLAUDE.md and project README
```

## Success Criteria

**Must Have (Blocking)**:
- ✓ All 16 non-Neo4j tests pass
- ✓ All 30 tests pass with Neo4j running
- ✓ Import works without [kg] dependencies (base classes)
- ✓ Helpful error message when Neo4j unavailable
- ✓ All ruff, mypy, isort checks pass
- ✓ CI passes (runs non-Neo4j tests)

**Should Have**:
- ✓ Complete .mdx documentation
- ✓ README.md in kg/ directory
- ✓ CLAUDE.md and project README updated

## Notes

- Preserves original PR structure (components/, graph_dbs/, requirements/, sampling/) for future Layer 5 expansion
- Neo4j tests skip gracefully in CI without NEO4J_URI
- Mock backend provides full testing capability without infrastructure
- Design maintains compatibility for future KGRag integration
