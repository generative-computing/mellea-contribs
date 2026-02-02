# Knowledge Graph (KG) Library

Backend-agnostic graph database abstraction for mellea-contribs.

## Installation

```bash
# Basic installation (includes MockGraphBackend)
pip install mellea-contribs

# With Neo4j support
pip install mellea-contribs[kg]
```

## Quick Start

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

### Layer 4: Backend Abstraction (Current)

- **GraphNode / GraphEdge / GraphPath**: Pure dataclasses representing graph data
- **GraphBackend**: Abstract interface for graph databases
- **Neo4jBackend**: Production-ready Neo4j implementation
- **MockGraphBackend**: Testing backend (no infrastructure required)

### Layer 2: Components (Planned)

- Full GraphQuery component with fluent builder pattern
- GraphResult with format_for_llm() and multiple output styles
- GraphTraversal with pattern matching

### Layer 3: LLM-Guided (Planned)

- Natural language to Cypher query conversion
- Query validation and repair loops
- Graph-specific requirements

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

## Status

- ✓ **Layer 4**: Backend abstraction (complete)
  - GraphNode, GraphEdge, GraphPath
  - GraphBackend interface
  - Neo4jBackend implementation
  - MockGraphBackend for testing

- ⧗ **Layer 2**: Full components (future)
- ⧗ **Layer 3**: LLM-guided queries (future)
- ⧗ **Layer 1**: Application integrations (future)

## Design Notes

- Pure dataclasses (GraphNode, GraphEdge, GraphPath) for data representation
- Components for queries and results (evolving in Layer 2)
- Async/await throughout for scalability
- Optional Neo4j dependency - graceful degradation if not installed
- MockBackend for unit testing without infrastructure

## See Also

- [Design Documentation](../../docs/kg_overview.mdx) - Detailed architecture and design
- [Mellea Framework](https://github.com/generative-computing/mellea) - Parent framework
