"""Knowledge Graph library for mellea-contribs.

Backend-agnostic graph database abstraction for graph-based RAG applications.

Optional Dependencies:
    Neo4j support requires: pip install mellea-contribs[kg]

Example:
    Basic usage with MockGraphBackend::

        from mellea_contribs.kg import MockGraphBackend, GraphNode

        backend = MockGraphBackend()
        node = GraphNode(id="1", label="Person", properties={"name": "Alice"})

    With Neo4j::

        from mellea_contribs.kg import Neo4jBackend

        backend = Neo4jBackend(
            uri="bolt://localhost:7687",
            auth=("neo4j", "password")
        )
"""

from mellea_contribs.kg.base import GraphEdge, GraphNode, GraphPath
from mellea_contribs.kg.graph_dbs.base import GraphBackend
from mellea_contribs.kg.graph_dbs.mock import MockGraphBackend

try:
    from mellea_contribs.kg.graph_dbs.neo4j import Neo4jBackend
except ImportError as e:

    def Neo4jBackend(*args, **kwargs):  # type: ignore[misc]
        """Raise ImportError with helpful message when Neo4j not installed."""
        raise ImportError(
            "Neo4j support requires additional dependencies. "
            "Install with: pip install mellea-contribs[kg]"
        ) from e


__all__ = [
    "GraphNode",
    "GraphEdge",
    "GraphPath",
    "GraphBackend",
    "Neo4jBackend",
    "MockGraphBackend",
]
