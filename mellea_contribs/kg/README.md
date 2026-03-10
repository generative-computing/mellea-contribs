# Knowledge Graph (KG) Library

A complete Knowledge Graph-Enhanced Retrieval-Augmented Generation (KG-RAG) system built with the Mellea framework. Combines multi-hop reasoning, entity extraction, consensus validation, and graph database backends for sophisticated question answering and knowledge graph updates.

## Overview

The KG library provides:

- **Multi-route QA Pipeline**: Break down complex questions into multiple solving routes and reach consensus
- **Document-based KG Updates**: Extract entities and relations from documents and merge with existing knowledge graphs
- **Backend-agnostic Design**: Works with Neo4j (production) or MockGraphBackend (testing)
- **LLM-Guided Operations**: All decisions powered by Mellea's @generative framework
- **Structured Data Models**: Pydantic models for all inputs and outputs

## Installation

```bash
# Basic installation (MockGraphBackend, no database)
pip install mellea-contribs

# With Neo4j support (for production)
pip install mellea-contribs[kg]

# With progress bars (tqdm)
pip install mellea-contribs[kg-utils]

# Complete installation (everything)
pip install mellea-contribs[kg,kg-utils,dev]
```

**Optional Dependencies:**
- `tqdm`: Progress bars for batch processing 
- `neo4j`: Neo4j driver for production backend 
- `rapidfuzz`: Fuzzy string matching for evaluation

## Quick Start

### Knowledge Graph-Enhanced Question Answering (Multi-Route QA)

```python
import asyncio
from mellea import start_session
from mellea_contribs.kg import (
    orchestrate_qa_retrieval,
    MockGraphBackend,
)

async def main():
    # Initialize Mellea session for LLM calls
    session = start_session(backend_name="litellm", model_id="gpt-4o-mini")

    # Use mock backend for testing (or Neo4jBackend for production)
    backend = MockGraphBackend()

    # Multi-route QA pipeline
    answer = await orchestrate_qa_retrieval(
        session=session,
        backend=backend,
        query="Who directed the highest-grossing film of 2024?",
        query_time="2024-12-31",
        domain="movies",
        num_routes=3,  # Explore 3 different reasoning paths
        hints="Consider box office revenue data",
    )

    print(f"Answer: {answer}")
    await backend.close()

asyncio.run(main())
```

**Pipeline Steps:**
1. Break down question into 3 solving routes
2. Extract topic entities from each route
3. Align entities with knowledge graph
4. Prune relevant relations
5. Evaluate if knowledge is sufficient
6. Validate consensus across routes
7. Return final answer with reasoning

### Document-Based Knowledge Graph Updates

```python
from mellea_contribs.kg import orchestrate_kg_update

async def update_kg():
    # Extract entities and relations from document
    result = await orchestrate_kg_update(
        session=session,
        backend=backend,
        doc_text="""
        Oppenheimer is a 2023 biographical film directed by Christopher Nolan.
        It stars Cillian Murphy and Emily Blunt. The film won Best Picture at
        the 2024 Academy Awards.
        """,
        domain="movies",
        entity_types="Person,Movie,Award",
        relation_types="DIRECTED,STARRED_IN,WON",
    )

    print(f"Extracted {len(result['extracted_entities'])} entities")
    print(f"Extracted {len(result['extracted_relations'])} relations")
    # Output: Automatically aligns and merges with existing KG data

asyncio.run(update_kg())
```

**Pipeline Steps:**
1. Extract entities and relations from text
2. Align extracted entities with existing KG entities
3. Decide whether to merge or create new entities
4. Align extracted relations with existing KG relations
5. Decide whether to merge or create new relations
6. Update knowledge graph with merged data

### Using Mock Backend (No Infrastructure)

```python
from mellea_contribs.kg import MockGraphBackend, GraphNode, GraphEdge

# Create mock nodes
alice = GraphNode(id="1", label="Person", properties={"name": "Alice"})
bob = GraphNode(id="2", label="Person", properties={"name": "Bob"})

# Create mock edge
knows = GraphEdge(
    id="e1",
    source=alice,
    label="KNOWS",
    target=bob,
    properties={}
)

# Create backend and execute query
backend = MockGraphBackend(
    mock_nodes=[alice, bob],
    mock_edges=[knows]
)

from mellea_contribs.kg import GraphQuery

query = GraphQuery(query_string="MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b")
result = await backend.execute_query(query)

print(f"Nodes: {len(result.nodes)}")
print(f"Edges: {len(result.edges)}")
```

### Using Neo4j Backend

```python
from mellea_contribs.kg import Neo4jBackend, GraphQuery

# Connect to Neo4j
backend = Neo4jBackend(
    connection_uri="bolt://localhost:7687",
    auth=("neo4j", "password")
)

# Execute Cypher query
query = GraphQuery(
    query_string="MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p, m",
    parameters={}
)

result = await backend.execute_query(query)

# Get schema
schema = await backend.get_schema()
print(f"Node types: {schema['node_types']}")
print(f"Edge types: {schema['edge_types']}")

# Validate query before execution
is_valid, error = await backend.validate_query(query)
print(f"Query valid: {is_valid}")

# Cleanup
await backend.close()
```

## Architecture

The system follows a 4-layer architecture:

### Layer 1: Application Orchestration

Entry points for KG-RAG pipelines:
- **`orchestrate_qa_retrieval()`**: Multi-route question answering with consensus validation
- **`orchestrate_kg_update()`**: Document-based KG updates with entity/relation alignment

### Layer 2: Components & Query Building

Query construction and result formatting:
- **GraphQuery / CypherQuery / SparqlQuery**: Query type abstractions
- **GraphResult**: Result formatting with `format_for_llm()` method
- **natural_language_to_cypher()**: Convert questions to Cypher queries
- **explain_query_result()**: Format results for LLM consumption

### Layer 3: LLM-Guided Logic (@generative functions)

All decisions made by LLM through Mellea's @generative framework:

**QA Functions (8):**
1. `break_down_question()` → Routes: Break complex questions into solving strategies
2. `extract_topic_entities()` → TopicEntities: Extract search entities from query
3. `align_topic_entities()` → RelevantEntities: Score entity relevance (0-1 scale)
4. `prune_relations()` → RelevantRelations: Filter relevant relations from entities
5. `prune_triplets()` → RelevantRelations: Score triplet relevance for answering
6. `evaluate_knowledge_sufficiency()` → EvaluationResult: Determine if KG knowledge suffices
7. `validate_consensus()` → ValidationResult: Validate consensus across routes
8. `generate_direct_answer()` → DirectAnswer: Generate answer without KG (fallback)

**Update Functions (5):**
1. `extract_entities_and_relations()` → ExtractionResult: Extract from documents
2. `align_entity_with_kg()` → AlignmentResult: Find matching KG entities
3. `decide_entity_merge()` → MergeDecision: Decide entity merge strategy
4. `align_relation_with_kg()` → AlignmentResult: Find matching KG relations
5. `decide_relation_merge()` → MergeDecision: Decide relation merge strategy

### Layer 4: Backend Abstraction

Database operations:
- **GraphNode / GraphEdge / GraphPath**: Pure dataclasses representing graph data
- **GraphBackend**: Abstract interface for graph databases
- **Neo4jBackend**: Production-ready Neo4j implementation
- **MockGraphBackend**: In-memory testing backend (no infrastructure required)

## Data Structures

### GraphNode

```python
@dataclass
class GraphNode:
    id: str                      # Unique identifier
    label: str                   # Node type/label
    properties: dict[str, Any]   # Node properties
```

### GraphEdge

```python
@dataclass
class GraphEdge:
    id: str                      # Unique identifier
    source: GraphNode            # Source node
    label: str                   # Relationship type
    target: GraphNode            # Target node
    properties: dict[str, Any]   # Relationship properties
```

### GraphPath

```python
@dataclass
class GraphPath:
    nodes: list[GraphNode]       # Sequence of nodes
    edges: list[GraphEdge]       # Sequence of edges
```

## Backend Interface

All backends implement `GraphBackend` which provides:

- `execute_query(query: GraphQuery) -> GraphResult`: Execute a query
- `get_schema() -> dict`: Get graph schema (node types, edge types, properties)
- `validate_query(query: GraphQuery) -> tuple[bool, str | None]`: Validate query
- `supports_query_type(query_type: str) -> bool`: Check if query type supported
- `execute_traversal(traversal: GraphTraversal) -> GraphResult`: Execute traversal pattern
- `close()`: Close backend connections

## Testing

```bash
# Run base data structure tests (no dependencies)
pytest test/kg/test_base.py -v

# Run mock backend tests (no dependencies)
pytest test/kg/test_mock_backend.py -v

# Run Neo4j tests (requires Neo4j running)
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password

pytest test/kg/test_neo4j_backend.py -v
```

## Starting Neo4j for Testing

```bash
# Docker
docker run -d --name neo4j-test -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/testpassword \
  neo4j:5.0

# Run tests
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=testpassword

pytest test/kg/ -v

# Cleanup
docker stop neo4j-test && docker rm neo4j-test
```

## Implementation Status

### Phase 1: Core KG Modules ✓ COMPLETE
- ✓ **Layer 1**: Application Orchestration
  - `orchestrate_qa_retrieval()` - Multi-route QA entry point
  - `orchestrate_kg_update()` - KG update entry point

- ✓ **Layer 2**: Components
  - GraphQuery, CypherQuery, SparqlQuery types
  - GraphResult with format_for_llm()
  - natural_language_to_cypher, explain_query_result

- ✓ **Layer 3**: LLM-Guided Logic
  - 8 QA @generative functions with full prompts
  - 5 Update @generative functions with full prompts
  - 12 Pydantic models for structured outputs

- ✓ **Layer 4**: Backend Abstraction
  - GraphNode, GraphEdge, GraphPath data structures
  - GraphBackend abstract interface
  - Neo4jBackend production implementation
  - MockGraphBackend for testing

### Phase 2: Run Scripts ✓ COMPLETE
- ✓ **8 Production-Ready CLI Scripts** (docs/examples/kgrag/scripts/)
  - Dataset creation: create_demo_dataset.py, create_tiny_dataset.py, create_truncated_dataset.py
  - Pipeline operations: run_kg_preprocess.py, run_kg_embed.py, run_kg_update.py, run_qa.py, run_eval.py
  - All scripts support --mock flag for testing without database
  - JSONL I/O for seamless pipeline chaining

### Phase 3: Utility Modules ✓ COMPLETE (95 Tests Passing)
- ✓ **5 Reusable Utility Modules** (mellea_contribs/kg/utils/)
  - `data_utils.py` - JSONL I/O, batching, schema validation (27 tests)
  - `session_manager.py` - Session/backend factories, async resource management (19 tests)
  - `progress.py` - Logging, progress tracking, JSON output (23 tests)
  - `eval_utils.py` - Evaluation metrics, result aggregation (26 tests)
  - All utilities tested with 95 comprehensive unit + integration tests

### Phase 4: Configuration & Validation ✓ COMPLETE
- ✓ **.env_template** - Configuration template with all variables
- ✓ **pyproject.toml** - Updated with kg-utils optional dependency group
- ✓ **sun.sh** - Comprehensive end-to-end test suite validating all phases

## Utility Modules (Phase 3)

The `mellea_contribs.kg.utils` package provides reusable utilities extracted from the run scripts:

### JSONL Data Utilities
```python
from mellea_contribs.kg.utils import (
    load_jsonl, save_jsonl, append_jsonl,
    batch_iterator, truncate_jsonl, shuffle_jsonl,
    validate_jsonl_schema
)

# Load JSONL file
items = list(load_jsonl("data/questions.jsonl"))

# Save and append
save_jsonl(items, "output/results.jsonl")
append_jsonl({"new": "item"}, "output/results.jsonl")

# Batch processing
batches = list(batch_iterator(items, batch_size=10))

# Truncate and shuffle
truncate_jsonl("input.jsonl", "output.jsonl", max_items=100)
shuffle_jsonl("input.jsonl", "output_shuffled.jsonl")

# Validate schema
valid, errors = validate_jsonl_schema("data.jsonl", required_fields=["id", "text"])
```

### Session & Backend Management
```python
from mellea_contribs.kg.utils import (
    create_session, create_backend, MelleaResourceManager
)

# Create session and backend
session = create_session(model_id="gpt-4o-mini")
backend = create_backend(backend_type="mock")

# Or use async context manager for automatic cleanup
async with MelleaResourceManager(backend_type="mock") as manager:
    # manager.session and manager.backend available
    schema = await manager.backend.get_schema()
```

### Progress Tracking & Logging
```python
from mellea_contribs.kg.utils import (
    setup_logging, log_progress, output_json,
    print_stats, ProgressTracker
)

# Setup logging
setup_logging(log_level="INFO", log_file="pipeline.log")
log_progress("Processing started", level="INFO")

# Output JSON
stats = compute_stats()
output_json(stats)  # Prints to stdout

# Print formatted stats
print_stats(stats, indent=2, to_stderr=False)

# Progress tracking
tracker = ProgressTracker(total=1000, desc="Processing")
for item in items:
    process(item)
    tracker.update(1)
tracker.close()
```

### Evaluation Metrics
```python
from mellea_contribs.kg.utils import (
    exact_match, fuzzy_match, mean_reciprocal_rank,
    precision, recall, f1_score,
    aggregate_qa_results, aggregate_update_results
)

# Matching
is_match = exact_match("Paris", "PARIS")  # True (case-insensitive)
is_similar = fuzzy_match("Oppenheimer", "Oppenheimer", threshold=0.8)  # True

# Metrics
mrr = mean_reciprocal_rank(qa_results)
prec = precision(predicted_entities, expected_entities)
rec = recall(predicted_entities, expected_entities)
f1 = f1_score(prec, rec)

# Aggregation
stats = aggregate_qa_results(qa_results_list)
stats = aggregate_update_results(update_results_list)
```

### Complete Workflow Example
```python
from mellea_contribs.kg.utils import (
    load_jsonl, batch_iterator, create_session, create_backend,
    log_progress, output_json, aggregate_qa_results
)

async def evaluate_qa_pipeline():
    # Setup
    setup_logging(log_level="INFO")
    session = create_session(model_id="gpt-4o-mini")
    backend = create_backend(backend_type="mock")

    # Load questions
    questions = list(load_jsonl("questions.jsonl"))

    # Process in batches
    all_results = []
    for batch in batch_iterator(questions, batch_size=10):
        results = await run_qa_batch(session, backend, batch)
        all_results.extend(results)

    # Aggregate and output
    stats = aggregate_qa_results(all_results)
    output_json(stats)

    await backend.close()
```

## Testing & Validation (Phase 3 & 4)

### Running Tests
```bash
# All KG tests
pytest test/kg/ -v

# Unit tests only (Phase 1, 3)
pytest test/kg/ --ignore=test/kg/test_scripts/ -v

# Utility module tests (95 tests)
pytest test/kg/utils/ -v

# Neo4j tests (requires running Neo4j)
export NEO4J_URI=bolt://localhost:7687
pytest test/kg/ -v -m neo4j
```

### Comprehensive Validation Suite (sun.sh)
```bash
# Run complete end-to-end validation
./sun.sh

# Quick validation (skip some slower tests)
./sun.sh --quick

# Unit tests only
./sun.sh --unit-only
```

The `sun.sh` script validates:
- Phase 0: Environment (Python, dependencies, imports)
- Phase 1: Core KG modules (95+ unit tests)
- Phase 2: Run scripts (all 8 scripts with mock backend)
- Phase 3: Utility modules (95 comprehensive tests)
- Phase 4: Configuration and dependencies

### Test Coverage
- **Phase 1**: Core modules (entity models, preprocessor, embedder, orchestrators)
- **Phase 2**: Run scripts (dataset creation, preprocessing, embedding, QA, evaluation)
- **Phase 3**: Utility modules
  - JSONL I/O: 27 tests (load, save, append, batch, truncate, shuffle, validate)
  - Session management: 19 tests (backend creation, session creation, async resources)
  - Progress/logging: 23 tests (logging levels, JSON output, progress tracking)
  - Evaluation: 26 tests (exact/fuzzy matching, MRR, precision/recall/F1, aggregation)

## Key Problems Solved

### Multi-Hop Reasoning
Traditional LLMs struggle with questions requiring multiple steps through a knowledge graph. KG-RAG breaks questions into solving routes and explores them systematically.

**Example:**
- Query: "Who won Best Picture at the Oscars and what other awards did they win?"
- Solved by: Entity extraction → Relation discovery → Multi-hop traversal → Consensus

### Temporal Understanding
Time-sensitive queries require proper context. The system tracks query times and considers temporal aspects in both questions and graph properties.

**Example:**
- Query: "Who was the highest-paid actor in 2023?" (different from 2024)
- Handled by: query_time parameter → temporal property filtering → time-aware alignment

### Structured Relationship Comprehension
Complex relationships with properties need careful reasoning. The system scores and filters relations based on relevance.

**Example:**
- Query: "Which movies did actor X star in that won awards?"
- Handled by: Extract ACTED_IN relations → Filter by WON properties → Score relevance

### Explainable Reasoning
Get not just answers, but reasoning paths through the knowledge graph showing how the answer was derived.

**Example:** Answer includes:
- Which solving route was used
- What entities were found
- Which relations were traversed
- Why the answer was sufficient or needed fallback

### Document Integration
Automatically extract new information from documents and intelligently merge with existing knowledge graph without duplicates.

**Example:** Merge "Leonardo DiCaprio" from document with existing entity, preserving both old and new properties.

## Performance Considerations

### Optimization Strategies

1. **Multi-Route Exploration**: Configurable number of solving routes
   - Fewer routes = faster but less certain
   - More routes = slower but more confident

2. **Relation Pruning Width**: Control how many relations to explore
   - Default: 20 relations per entity
   - Adjustable via `width` parameter

3. **Consensus Validation**: Stop early when routes agree
   - Fast path: 2 of 3 routes agree → return answer
   - Slow path: All routes explored → reach consensus

4. **Caching**: Neo4j vector index caching, schema caching
   - Results cached per query_time + domain combination
   - Entity similarity searches cached by backend

### Scalability

- **Async/await throughout** for non-blocking I/O
- **Configurable parameters** for tuning vs. quality tradeoff
- **Efficient Pydantic models** for structured validation
- **MockBackend** for parallel testing without infrastructure

## Known Limitations

- **Domain-Specific**: Currently optimized for movie/entertainment domain (easily adapted)
- **Requires Pre-built Graphs**: Expects Neo4j or data in MockBackend already populated
- **Computational Cost**: Multi-hop traversal can be expensive on large graphs
- **English-Only**: Currently designed for English-language queries (LLM-dependent)
- **Entity Disambiguation**: Relies on good entity naming conventions in KG

## Design Notes

- Pure dataclasses (GraphNode, GraphEdge, GraphPath) for data representation
- Components for queries and results (evolving in Layer 2)
- Async/await throughout for scalability
- Optional Neo4j dependency - graceful degradation if not installed
- MockBackend for unit testing without infrastructure
- All LLM decisions through Mellea's @generative framework (pluggable LLM backends)
- Structured outputs via Pydantic models (validated at LLM output)

## Migration Notes

This library was adapted from the KGRag system in mellea PR#3. Key differences:

- **Open-ended**: Works with any domain (not movie-specific)
- **Mellea-integrated**: Uses Mellea's @generative decorators and MelleaSession
- **Backend-agnostic**: MockBackend for testing, Neo4jBackend for production
- **Structured API**: Clear Layer 1-4 separation with orchestration entry points
- **Full type hints**: Pydantic models throughout

## Quick Reference: Running Everything

### Minimal Setup (Mock Backend, No Database)
```bash
# Install
pip install -e .[kg,kg-utils]

# Run tests
pytest test/kg/utils/ -v

# Run full validation
./sun.sh

# Try an example script
python docs/examples/kgrag/scripts/create_demo_dataset.py --output /tmp/demo.jsonl
python docs/examples/kgrag/scripts/run_qa.py --input /tmp/demo.jsonl --mock --output /tmp/qa.jsonl
```

### Production Setup (With Neo4j)
```bash
# Install with all features
pip install -e .[kg,kg-utils,dev]

# Start Neo4j
docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:5.0

# Configure
cp .env_template .env
# Edit .env with your Neo4j credentials

# Run scripts
python docs/examples/kgrag/scripts/run_kg_preprocess.py --input data.jsonl
python docs/examples/kgrag/scripts/run_qa.py --input questions.jsonl --output results.jsonl
```

## See Also

- [Main README](../../README.md) - KG-RAG overview and quick start
- [CLAUDE.md](../../CLAUDE.md) - Development guide and architecture
- [PHASE4_CONFIGURATION.md](../../PHASE4_CONFIGURATION.md) - Configuration templates and setup
- [missing_for_run_sh.txt](../../missing_for_run_sh.txt) - Implementation status and progress
- [Test README](utils/README.md) - Phase 3 utility module tests documentation
- [Mellea Framework](https://github.com/generative-computing/mellea) - Parent framework
- [Original PR#3](https://github.com/ydzhu98/mellea/pull/3) - Source of KG-RAG system
