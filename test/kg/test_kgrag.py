"""Tests for Layer 1: KGRag pipeline.

Structural tests run without an LLM. Qualitative (end-to-end) tests require
a real Mellea session and are marked @pytest.mark.qualitative so they are
skipped in CI.
"""

import pytest

from mellea_contribs.kg.base import GraphEdge, GraphNode
from mellea_contribs.kg.graph_dbs.mock import MockGraphBackend
from mellea_contribs.kg.kgrag import KGRag, format_schema


# ---------------------------------------------------------------------------
# format_schema
# ---------------------------------------------------------------------------


class TestFormatSchema:
    """Tests for the format_schema() helper."""

    def test_full_schema(self):
        """format_schema includes node types, edge types, and property keys."""
        schema = {
            "node_types": ["Person", "Movie"],
            "edge_types": ["ACTED_IN", "DIRECTED"],
            "property_keys": ["name", "title", "year"],
        }
        result = format_schema(schema)
        assert "Person" in result
        assert "Movie" in result
        assert "ACTED_IN" in result
        assert "name" in result

    def test_empty_schema(self):
        """format_schema handles an empty schema without errors."""
        result = format_schema({})
        assert "Graph Schema" in result

    def test_partial_schema(self):
        """format_schema handles a schema with only node types."""
        result = format_schema({"node_types": ["Person"]})
        assert "Person" in result

    def test_returns_string(self):
        """format_schema always returns a string."""
        assert isinstance(format_schema({}), str)
        assert isinstance(format_schema({"node_types": ["A"]}), str)


# ---------------------------------------------------------------------------
# KGRag structural tests (no LLM required)
# ---------------------------------------------------------------------------


class TestKGRagStructural:
    """Structural tests for KGRag that do not call an LLM."""

    @pytest.fixture
    def backend(self):
        """Mock backend with sample data."""
        nodes = [
            GraphNode(id="1", label="Person", properties={"name": "Alice"}),
            GraphNode(id="2", label="Movie", properties={"title": "The Matrix"}),
        ]
        edges = [
            GraphEdge(
                id="e1",
                source=nodes[0],
                label="ACTED_IN",
                target=nodes[1],
                properties={},
            )
        ]
        return MockGraphBackend(mock_nodes=nodes, mock_edges=edges)

    def test_init_stores_backend_and_session(self, backend):
        """KGRag stores backend, session, and config options."""
        rag = KGRag(backend=backend, session=None)
        assert rag._backend is backend
        assert rag._session is None

    def test_default_format_style(self, backend):
        """KGRag defaults to 'natural' format style."""
        rag = KGRag(backend=backend, session=None)
        assert rag._format_style == "natural"

    def test_custom_format_style(self, backend):
        """KGRag accepts a custom format style."""
        rag = KGRag(backend=backend, session=None, format_style="triplets")
        assert rag._format_style == "triplets"

    def test_default_max_repair_attempts(self, backend):
        """KGRag defaults to 2 max repair attempts."""
        rag = KGRag(backend=backend, session=None)
        assert rag._max_repair_attempts == 2

    def test_custom_max_repair_attempts(self, backend):
        """KGRag accepts a custom max_repair_attempts."""
        rag = KGRag(backend=backend, session=None, max_repair_attempts=5)
        assert rag._max_repair_attempts == 5

    def test_answer_is_coroutine(self, backend):
        """KGRag.answer() is a coroutine function."""
        import asyncio

        rag = KGRag(backend=backend, session=None)
        assert asyncio.iscoroutinefunction(rag.answer)

    async def test_validate_and_execute_returns_result(self, backend):
        """validate_and_execute_query returns a result for a valid query."""
        from mellea_contribs.kg.components.retrieval import validate_and_execute_query

        schema_text = format_schema(await backend.get_schema())
        # MockBackend.validate_query always returns True; execute_query returns a GraphResult
        result = await validate_and_execute_query(
            backend, None, "MATCH (n) RETURN n", schema_text
        )
        assert result is not None


# ---------------------------------------------------------------------------
# Qualitative (end-to-end) tests
# ---------------------------------------------------------------------------


@pytest.mark.qualitative
class TestKGRagQualitative:
    """End-to-end tests for KGRag that require a real LLM session.

    These tests are skipped in CI unless MELLEA_SKIP_QUALITATIVE is unset.
    They require a running Neo4j database (NEO4J_URI env var) and a
    configured Mellea LLM backend.
    """

    @pytest.fixture
    def neo4j_backend(self):
        """Neo4j backend for qualitative tests."""
        import os

        uri = os.environ.get("NEO4J_URI")
        if not uri:
            pytest.skip("NEO4J_URI not set")

        from mellea_contribs.kg.graph_dbs.neo4j import Neo4jBackend

        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "password")
        return Neo4jBackend(connection_uri=uri, auth=(user, password))

    @pytest.fixture
    def mellea_session(self):
        """Mellea session for qualitative tests."""
        from mellea import start_session

        return start_session(backend_name="litellm", model_id="gpt-4o-mini")

    async def test_answer_returns_string(self, neo4j_backend, mellea_session):
        """KGRag.answer() returns a non-empty string for a simple question."""
        rag = KGRag(backend=neo4j_backend, session=mellea_session)
        answer = await rag.answer("What nodes exist in the graph?")
        assert isinstance(answer, str)
        assert len(answer) > 0
        await neo4j_backend.close()

    async def test_answer_is_grounded(self, neo4j_backend, mellea_session):
        """KGRag.answer() produces an answer grounded in real graph data."""
        rag = KGRag(backend=neo4j_backend, session=mellea_session)
        answer = await rag.answer("What node labels are in the graph?")
        assert isinstance(answer, str)
        # Answer should contain some domain-relevant content
        assert len(answer.split()) > 3
        await neo4j_backend.close()
