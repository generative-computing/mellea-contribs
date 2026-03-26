"""Tests for Layer 4: MockGraphBackend."""

import pytest

from mellea_contribs.kg.base import GraphEdge, GraphNode
from mellea_contribs.kg.graph_dbs.mock import MockGraphBackend


@pytest.fixture
def mock_nodes():
    """Create mock nodes for testing."""
    return [
        GraphNode(id="1", label="Person", properties={"name": "Alice"}),
        GraphNode(id="2", label="Person", properties={"name": "Bob"}),
        GraphNode(id="3", label="Movie", properties={"title": "The Matrix"}),
    ]


@pytest.fixture
def mock_edges(mock_nodes):
    """Create mock edges for testing."""
    return [
        GraphEdge(
            id="e1",
            source=mock_nodes[0],
            label="ACTED_IN",
            target=mock_nodes[2],
            properties={"role": "Neo"},
        ),
        GraphEdge(
            id="e2",
            source=mock_nodes[1],
            label="ACTED_IN",
            target=mock_nodes[2],
            properties={"role": "Morpheus"},
        ),
    ]


@pytest.fixture
def mock_backend(mock_nodes, mock_edges):
    """Create a mock backend for testing."""
    return MockGraphBackend(mock_nodes=mock_nodes, mock_edges=mock_edges)


class TestMockGraphBackend:
    """Tests for Layer 4: MockGraphBackend."""

    def test_create_mock_backend(self):
        """Test creating a MockGraphBackend."""
        backend = MockGraphBackend()

        assert backend.backend_id == "mock"
        assert backend.connection_uri == "mock://localhost"
        assert backend.mock_nodes == []
        assert backend.mock_edges == []

    def test_mock_backend_with_data(self, mock_backend, mock_nodes, mock_edges):
        """Test MockGraphBackend with predefined data."""
        assert len(mock_backend.mock_nodes) == 3
        assert len(mock_backend.mock_edges) == 2
        assert mock_backend.mock_nodes == mock_nodes
        assert mock_backend.mock_edges == mock_edges

    @pytest.mark.asyncio
    async def test_get_schema(self, mock_backend):
        """Test getting mock schema."""
        schema = await mock_backend.get_schema()

        assert "node_types" in schema
        assert "edge_types" in schema
        assert "property_keys" in schema
        assert "MockNode" in schema["node_types"]
        assert "MOCK_EDGE" in schema["edge_types"]

    @pytest.mark.asyncio
    async def test_validate_query_always_valid(self, mock_backend):
        """Test that mock backend always validates queries as valid."""
        # Need to create a minimal query object
        from mellea_contribs.kg.components.query import GraphQuery

        query = GraphQuery(query_string="MATCH (n) RETURN n")
        is_valid, error = await mock_backend.validate_query(query)

        assert is_valid is True
        assert error is None

    def test_supports_all_query_types(self, mock_backend):
        """Test that mock backend supports all query types."""
        assert mock_backend.supports_query_type("cypher") is True
        assert mock_backend.supports_query_type("sparql") is True
        assert mock_backend.supports_query_type("gremlin") is True
        assert mock_backend.supports_query_type("unknown") is True

    def test_query_history_tracking(self, mock_backend):
        """Test that mock backend tracks query history."""
        assert len(mock_backend.query_history) == 0

        mock_backend.clear_history()
        assert len(mock_backend.query_history) == 0

    def test_clear_history(self, mock_backend):
        """Test clearing query history."""
        mock_backend.query_history.append(("MATCH (n) RETURN n", {}))
        assert len(mock_backend.query_history) == 1

        mock_backend.clear_history()
        assert len(mock_backend.query_history) == 0
