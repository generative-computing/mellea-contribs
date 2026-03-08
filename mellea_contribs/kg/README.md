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
# Basic installation (includes MockGraphBackend)
pip install mellea-contribs

# With Neo4j support
pip install mellea-contribs[kg]
```

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

- ✓ **Layer 1**: Application Orchestration (complete)
  - `orchestrate_qa_retrieval()` - Multi-route QA entry point
  - `orchestrate_kg_update()` - KG update entry point

- ✓ **Layer 2**: Components (partial)
  - GraphQuery, CypherQuery, SparqlQuery types
  - GraphResult with format_for_llm()
  - natural_language_to_cypher, explain_query_result
  - [Placeholder: Layer 2 controller functions for full orchestration]

- ✓ **Layer 3**: LLM-Guided Logic (complete)
  - 8 QA @generative functions with full prompts
  - 5 Update @generative functions with full prompts
  - 12 Pydantic models for structured outputs
  - [Placeholder: Layer 3 helper utilities for parsing/formatting]

- ✓ **Layer 4**: Backend Abstraction (complete)
  - GraphNode, GraphEdge, GraphPath data structures
  - GraphBackend abstract interface
  - Neo4jBackend production implementation
  - MockGraphBackend for testing

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

## See Also

- [Main README](../../README.md) - KG-RAG overview and quick start
- [CLAUDE.md](../../CLAUDE.md) - Development guide and architecture
- [Mellea Framework](https://github.com/generative-computing/mellea) - Parent framework
- [Original PR#3](https://github.com/ydzhu98/mellea/pull/3) - Source of KG-RAG system
