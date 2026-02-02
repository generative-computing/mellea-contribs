"""Core data structures for graph representation.

These are pure dataclasses, not Components. They represent graph data.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class GraphNode:
    """A node in a graph.

    This is a dataclass, not a Component. It's just data.
    """

    id: str
    label: str  # Node type/label
    properties: dict[str, Any]

    @classmethod
    def from_neo4j_node(cls, node: Any) -> "GraphNode":
        """Create from Neo4j node object.

        Args:
            node: Neo4j node object

        Returns:
            GraphNode instance
        """
        return cls(
            id=str(node.element_id),
            label=next(iter(node.labels)) if node.labels else "Unknown",
            properties=dict(node.items()),
        )


@dataclass
class GraphEdge:
    """An edge in a graph.

    This is a dataclass, not a Component. It's just data.
    """

    id: str
    source: GraphNode
    label: str  # Relationship type
    target: GraphNode
    properties: dict[str, Any]

    @classmethod
    def from_neo4j_relationship(
        cls, rel: Any, source: GraphNode, target: GraphNode
    ) -> "GraphEdge":
        """Create from Neo4j relationship object.

        Args:
            rel: Neo4j relationship object
            source: Source GraphNode
            target: Target GraphNode

        Returns:
            GraphEdge instance
        """
        return cls(
            id=str(rel.element_id),
            source=source,
            label=rel.type,
            target=target,
            properties=dict(rel.items()),
        )


@dataclass
class GraphPath:
    """A path through a graph (sequence of nodes and edges).

    This is a dataclass, not a Component. It's just data.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]

    @classmethod
    def from_neo4j_path(cls, path: Any) -> "GraphPath":
        """Create from Neo4j path object.

        Args:
            path: Neo4j path object

        Returns:
            GraphPath instance
        """
        nodes = [GraphNode.from_neo4j_node(node) for node in path.nodes]
        edges = []

        # Build edges from relationships
        for i, rel in enumerate(path.relationships):
            source = nodes[i]
            target = nodes[i + 1]
            edges.append(GraphEdge.from_neo4j_relationship(rel, source, target))

        return cls(nodes=nodes, edges=edges)
