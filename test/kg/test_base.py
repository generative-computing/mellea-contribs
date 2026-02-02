"""Tests for base graph data structures."""

import pytest

from mellea_contribs.kg.base import GraphEdge, GraphNode, GraphPath


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    def test_create_graph_node(self):
        """Test creating a GraphNode."""
        node = GraphNode(
            id="1", label="Person", properties={"name": "Alice", "age": 30}
        )

        assert node.id == "1"
        assert node.label == "Person"
        assert node.properties["name"] == "Alice"
        assert node.properties["age"] == 30

    def test_graph_node_empty_properties(self):
        """Test GraphNode with empty properties."""
        node = GraphNode(id="2", label="Movie", properties={})

        assert node.id == "2"
        assert node.label == "Movie"
        assert node.properties == {}

    def test_graph_node_equality(self):
        """Test GraphNode equality."""
        node1 = GraphNode(id="1", label="Person", properties={"name": "Alice"})
        node2 = GraphNode(id="1", label="Person", properties={"name": "Alice"})
        node3 = GraphNode(id="2", label="Person", properties={"name": "Bob"})

        assert node1 == node2
        assert node1 != node3


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    def test_create_graph_edge(self):
        """Test creating a GraphEdge."""
        source = GraphNode(id="1", label="Person", properties={"name": "Alice"})
        target = GraphNode(id="2", label="Movie", properties={"title": "The Matrix"})

        edge = GraphEdge(
            id="e1",
            source=source,
            label="ACTED_IN",
            target=target,
            properties={"role": "Neo"},
        )

        assert edge.id == "e1"
        assert edge.source == source
        assert edge.label == "ACTED_IN"
        assert edge.target == target
        assert edge.properties["role"] == "Neo"

    def test_graph_edge_empty_properties(self):
        """Test GraphEdge with empty properties."""
        source = GraphNode(id="1", label="Person", properties={})
        target = GraphNode(id="2", label="Movie", properties={})

        edge = GraphEdge(
            id="e1", source=source, label="DIRECTED", target=target, properties={}
        )

        assert edge.properties == {}

    def test_graph_edge_equality(self):
        """Test GraphEdge equality."""
        source = GraphNode(id="1", label="Person", properties={"name": "Alice"})
        target = GraphNode(id="2", label="Movie", properties={"title": "The Matrix"})

        edge1 = GraphEdge(
            id="e1", source=source, label="ACTED_IN", target=target, properties={}
        )
        edge2 = GraphEdge(
            id="e1", source=source, label="ACTED_IN", target=target, properties={}
        )
        edge3 = GraphEdge(
            id="e2", source=source, label="DIRECTED", target=target, properties={}
        )

        assert edge1 == edge2
        assert edge1 != edge3


class TestGraphPath:
    """Tests for GraphPath dataclass."""

    def test_create_graph_path(self):
        """Test creating a GraphPath."""
        node1 = GraphNode(id="1", label="Person", properties={"name": "Alice"})
        node2 = GraphNode(id="2", label="Movie", properties={"title": "The Matrix"})
        node3 = GraphNode(id="3", label="Person", properties={"name": "Bob"})

        edge1 = GraphEdge(
            id="e1", source=node1, label="ACTED_IN", target=node2, properties={}
        )
        edge2 = GraphEdge(
            id="e2", source=node3, label="ACTED_IN", target=node2, properties={}
        )

        path = GraphPath(nodes=[node1, node2, node3], edges=[edge1, edge2])

        assert len(path.nodes) == 3
        assert len(path.edges) == 2
        assert path.nodes[0] == node1
        assert path.nodes[1] == node2
        assert path.nodes[2] == node3
        assert path.edges[0] == edge1
        assert path.edges[1] == edge2

    def test_empty_graph_path(self):
        """Test creating an empty GraphPath."""
        path = GraphPath(nodes=[], edges=[])

        assert len(path.nodes) == 0
        assert len(path.edges) == 0

    def test_graph_path_single_node(self):
        """Test GraphPath with single node."""
        node = GraphNode(id="1", label="Person", properties={"name": "Alice"})
        path = GraphPath(nodes=[node], edges=[])

        assert len(path.nodes) == 1
        assert len(path.edges) == 0
        assert path.nodes[0] == node
