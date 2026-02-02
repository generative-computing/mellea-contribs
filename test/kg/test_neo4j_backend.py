"""Tests for Neo4jBackend.

These tests require a running Neo4j instance. They will be skipped if:
- Neo4j is not installed
- No Neo4j instance is available at the connection URI
- Authentication fails

To run these tests:

1. Set environment variables (recommended):
    export NEO4J_PASSWORD="your_password"
    uv run pytest test/contribs/kg/test_neo4j_backend.py -v

2. Or start Neo4j with Docker:
    docker run --rm -p 7687:7687 -p 7474:7474 \
        -e NEO4J_AUTH=neo4j/testpassword \
        neo4j:latest

Note: Fixtures are defined in conftest.py and use environment variables:
    - NEO4J_URI (default: bolt://localhost:7687)
    - NEO4J_USER (default: neo4j)
    - NEO4J_PASSWORD (default: testpassword)
"""

import os

import pytest

from mellea_contribs.kg.components.query import GraphQuery

try:
    from mellea_contribs.kg.graph_dbs.neo4j import Neo4jBackend

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


# Skip all tests in this module if Neo4j is not available
pytestmark = [
    pytest.mark.neo4j,
    pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Neo4j driver not installed"),
]


class TestNeo4jBackend:
    """Tests for Neo4jBackend."""

    @pytest.mark.asyncio
    async def test_create_neo4j_backend(self, neo4j_credentials):
        """Test creating a Neo4jBackend."""
        backend = Neo4jBackend(
            connection_uri=neo4j_credentials["uri"],
            auth=(neo4j_credentials["user"], neo4j_credentials["password"]),
        )

        assert backend.backend_id == "neo4j"
        assert backend.connection_uri == neo4j_credentials["uri"]
        assert backend.auth == (
            neo4j_credentials["user"],
            neo4j_credentials["password"],
        )

        await backend.close()

    @pytest.mark.asyncio
    async def test_supports_cypher(self, neo4j_backend):
        """Test that Neo4j backend supports Cypher queries."""
        assert neo4j_backend.supports_query_type("cypher") is True
        assert neo4j_backend.supports_query_type("sparql") is False

    @pytest.mark.asyncio
    async def test_get_schema(self, neo4j_backend):
        """Test getting Neo4j schema."""
        schema = await neo4j_backend.get_schema()

        assert "node_types" in schema
        assert "edge_types" in schema
        assert "property_keys" in schema
        assert isinstance(schema["node_types"], list)
        assert isinstance(schema["edge_types"], list)
        assert isinstance(schema["property_keys"], list)

    @pytest.mark.asyncio
    async def test_validate_valid_query(self, neo4j_backend):
        """Test validating a valid Cypher query."""
        query = GraphQuery(query_string="MATCH (n) RETURN n LIMIT 10")
        is_valid, error = await neo4j_backend.validate_query(query)

        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_validate_invalid_query(self, neo4j_backend):
        """Test validating an invalid Cypher query."""
        query = GraphQuery(query_string="INVALID CYPHER QUERY")
        is_valid, error = await neo4j_backend.validate_query(query)

        assert is_valid is False
        assert error is not None
        assert len(error) > 0

    @pytest.mark.asyncio
    async def test_execute_simple_query(self, populated_neo4j_backend):
        """Test executing a simple query."""
        query = GraphQuery(query_string="MATCH (p:Person) RETURN p ORDER BY p.name")
        result = await populated_neo4j_backend.execute_query(query)

        assert len(result.nodes) == 2
        assert result.nodes[0].label == "Person"
        assert result.nodes[0].properties["name"] == "Alice"
        assert result.nodes[1].properties["name"] == "Bob"

    @pytest.mark.asyncio
    async def test_execute_query_with_parameters(self, populated_neo4j_backend):
        """Test executing a query with parameters."""
        query = GraphQuery(
            query_string="MATCH (p:Person {name: $name}) RETURN p",
            parameters={"name": "Alice"},
        )
        result = await populated_neo4j_backend.execute_query(query)

        assert len(result.nodes) == 1
        assert result.nodes[0].properties["name"] == "Alice"
        assert result.nodes[0].properties["age"] == 30

    @pytest.mark.asyncio
    async def test_execute_query_with_relationships(self, populated_neo4j_backend):
        """Test executing a query that returns relationships."""
        query = GraphQuery(
            query_string="""
            MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
            RETURN p, r, m
            ORDER BY p.name
            """
        )
        result = await populated_neo4j_backend.execute_query(query)

        # Should have 2 people and 1 movie
        assert len(result.nodes) == 3
        assert len(result.edges) == 2

        # Check edge properties
        assert result.edges[0].label == "ACTED_IN"
        assert "role" in result.edges[0].properties

    @pytest.mark.asyncio
    async def test_execute_query_no_results(self, populated_neo4j_backend):
        """Test executing a query that returns no results."""
        query = GraphQuery(
            query_string="MATCH (p:Person {name: 'NonExistent'}) RETURN p"
        )
        result = await populated_neo4j_backend.execute_query(query)

        assert len(result.nodes) == 0
        assert len(result.edges) == 0

    @pytest.mark.asyncio
    async def test_execute_query_different_format_styles(self, populated_neo4j_backend):
        """Test executing query with different format styles."""
        query = GraphQuery(query_string="MATCH (p:Person) RETURN p")

        result_triplets = await populated_neo4j_backend.execute_query(
            query, format_style="triplets"
        )
        assert result_triplets.format_style == "triplets"

        result_natural = await populated_neo4j_backend.execute_query(
            query, format_style="natural"
        )
        assert result_natural.format_style == "natural"

    @pytest.mark.asyncio
    async def test_parse_neo4j_result_deduplication(self, populated_neo4j_backend):
        """Test that parsed results deduplicate nodes and edges."""
        query = GraphQuery(
            query_string="""
            MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
            RETURN p, r, m
            UNION
            MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
            RETURN p, r, m
            """
        )
        result = await populated_neo4j_backend.execute_query(query)

        # Should still have only 3 unique nodes and 2 unique edges despite UNION
        assert len(result.nodes) == 3
        assert len(result.edges) == 2

    @pytest.mark.asyncio
    async def test_execute_query_with_path(self, populated_neo4j_backend):
        """Test executing a query that returns a path."""
        query = GraphQuery(
            query_string="""
            MATCH path = (p:Person)-[:ACTED_IN]->(m:Movie)
            WHERE p.name = 'Alice'
            RETURN path
            """
        )
        result = await populated_neo4j_backend.execute_query(query)

        assert len(result.paths) == 1
        path = result.paths[0]
        assert len(path.nodes) == 2
        assert len(path.edges) == 1
        assert path.nodes[0].properties["name"] == "Alice"
        assert path.nodes[1].properties["title"] == "The Matrix"

    @pytest.mark.asyncio
    async def test_execute_query_empty_string(self, neo4j_backend):
        """Test that empty query string raises ValueError."""
        query = GraphQuery(query_string="")

        with pytest.raises(ValueError, match="Query string is empty"):
            await neo4j_backend.execute_query(query)

    @pytest.mark.asyncio
    async def test_backend_close(self, neo4j_credentials):
        """Test closing backend connections."""
        backend = Neo4jBackend(
            connection_uri=neo4j_credentials["uri"],
            auth=(neo4j_credentials["user"], neo4j_credentials["password"]),
        )

        # Should not raise
        await backend.close()
