"""Tests for Layer 3: Graph Query Components.

Covers:
- components/query.py: GraphQuery, CypherQuery, SparqlQuery
- components/result.py: GraphResult (all format styles)
- components/traversal.py: GraphTraversal (all patterns + to_cypher)
"""

import json

import pytest
from mellea.stdlib.components import Component, TemplateRepresentation

from mellea_contribs.kg.base import GraphEdge, GraphNode, GraphPath
from mellea_contribs.kg.components.query import CypherQuery, GraphQuery, SparqlQuery
from mellea_contribs.kg.components.result import GraphResult
from mellea_contribs.kg.components.traversal import GraphTraversal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def alice():
    return GraphNode(id="1", label="Person", properties={"name": "Alice"})


@pytest.fixture
def matrix():
    return GraphNode(id="2", label="Movie", properties={"title": "The Matrix"})


@pytest.fixture
def acted_in(alice, matrix):
    return GraphEdge(
        id="e1", source=alice, label="ACTED_IN", target=matrix, properties={}
    )


@pytest.fixture
def graph_path(alice, acted_in, matrix):
    return GraphPath(nodes=[alice, matrix], edges=[acted_in])


# ---------------------------------------------------------------------------
# GraphQuery
# ---------------------------------------------------------------------------


class TestGraphQuery:
    """Tests for the full GraphQuery Component."""

    def test_implements_component_protocol(self):
        """GraphQuery satisfies the Mellea Component protocol."""
        q = GraphQuery(query_string="MATCH (n) RETURN n")
        assert isinstance(q, Component)

    def test_properties_accessible(self):
        """GraphQuery exposes query_string, parameters, description, metadata."""
        q = GraphQuery(
            query_string="MATCH (n) RETURN n",
            parameters={"limit": 5},
            description="All nodes",
            metadata={"schema": "test"},
        )
        assert q.query_string == "MATCH (n) RETURN n"
        assert q.parameters == {"limit": 5}
        assert q.description == "All nodes"
        assert q.metadata == {"schema": "test"}

    def test_defaults(self):
        """GraphQuery defaults to empty parameters / metadata and None strings."""
        q = GraphQuery()
        assert q.query_string is None
        assert q.parameters == {}
        assert q.description is None
        assert q.metadata == {}

    def test_format_for_llm_returns_template_representation(self):
        """format_for_llm() returns a TemplateRepresentation."""
        q = GraphQuery(query_string="MATCH (n) RETURN n", description="All nodes")
        rep = q.format_for_llm()
        assert isinstance(rep, TemplateRepresentation)
        assert rep.args["query"] == "MATCH (n) RETURN n"
        assert rep.args["description"] == "All nodes"

    def test_with_description_is_immutable(self):
        """with_description() returns a new instance without modifying the original."""
        q = GraphQuery(query_string="MATCH (n) RETURN n")
        q2 = q.with_description("Updated")
        assert q.description is None
        assert q2.description == "Updated"
        assert q is not q2

    def test_with_parameters_merges(self):
        """with_parameters() merges new params into a new instance."""
        q = GraphQuery(parameters={"a": 1})
        q2 = q.with_parameters(b=2)
        assert q.parameters == {"a": 1}
        assert q2.parameters == {"a": 1, "b": 2}

    def test_with_metadata_merges(self):
        """with_metadata() merges new metadata into a new instance."""
        q = GraphQuery(metadata={"x": 1})
        q2 = q.with_metadata(y=2)
        assert q.metadata == {"x": 1}
        assert q2.metadata == {"x": 1, "y": 2}

    def test_parts_raises(self):
        """parts() raises NotImplementedError."""
        q = GraphQuery()
        with pytest.raises(NotImplementedError):
            q.parts()


# ---------------------------------------------------------------------------
# CypherQuery
# ---------------------------------------------------------------------------


class TestCypherQuery:
    """Tests for the fluent CypherQuery builder."""

    def test_is_graph_query(self):
        """CypherQuery is a subclass of GraphQuery."""
        assert issubclass(CypherQuery, GraphQuery)

    def test_empty_query(self):
        """CypherQuery with no clauses has None query_string."""
        q = CypherQuery()
        assert q.query_string is None

    def test_match_clause(self):
        """match() adds a MATCH clause."""
        q = CypherQuery().match("(n:Person)")
        assert q.query_string == "MATCH (n:Person)"

    def test_where_clause(self):
        """where() adds a WHERE condition."""
        q = CypherQuery().match("(n:Person)").where("n.age > 18")
        assert "WHERE n.age > 18" in q.query_string

    def test_return_clause(self):
        """return_() adds RETURN expressions."""
        q = CypherQuery().match("(n)").return_("n.name", "n.age")
        assert "RETURN n.name, n.age" in q.query_string

    def test_order_by_clause(self):
        """order_by() adds an ORDER BY clause."""
        q = CypherQuery().match("(n)").return_("n").order_by("n.name ASC")
        assert "ORDER BY n.name ASC" in q.query_string

    def test_limit_clause(self):
        """limit() adds a LIMIT clause."""
        q = CypherQuery().match("(n)").return_("n").limit(10)
        assert "LIMIT 10" in q.query_string

    def test_full_fluent_chain(self):
        """All builder methods compose correctly."""
        q = (
            CypherQuery()
            .match("(m:Movie)")
            .where("m.year = $year")
            .return_("m.title", "m.year")
            .order_by("m.year DESC")
            .limit(5)
            .with_parameters(year=2020)
            .with_description("Movies from 2020")
        )
        qs = q.query_string
        assert "MATCH (m:Movie)" in qs
        assert "WHERE m.year = $year" in qs
        assert "RETURN m.title, m.year" in qs
        assert "ORDER BY m.year DESC" in qs
        assert "LIMIT 5" in qs
        assert q.parameters == {"year": 2020}
        assert q.description == "Movies from 2020"

    def test_each_step_immutable(self):
        """Each builder step returns a new CypherQuery instance."""
        base = CypherQuery()
        with_match = base.match("(n)")
        assert base is not with_match
        assert base.query_string is None

    def test_multiple_where_conditions(self):
        """Multiple where() calls are ANDed together."""
        q = CypherQuery().match("(n)").where("n.age > 18").where("n.active = true")
        assert "n.age > 18 AND n.active = true" in q.query_string

    def test_explicit_query_string_bypasses_clauses(self):
        """Providing query_string directly bypasses clause building."""
        raw = "MATCH (n:Custom) RETURN n"
        q = CypherQuery(query_string=raw)
        assert q.query_string == raw

    def test_format_for_llm_includes_query_type(self):
        """CypherQuery format_for_llm includes query_type field."""
        q = CypherQuery(query_string="MATCH (n) RETURN n")
        rep = q.format_for_llm()
        assert rep.args["query_type"] == "Cypher (Neo4j)"


# ---------------------------------------------------------------------------
# SparqlQuery
# ---------------------------------------------------------------------------


class TestSparqlQuery:
    """Tests for SparqlQuery."""

    def test_is_graph_query(self):
        """SparqlQuery is a subclass of GraphQuery."""
        assert issubclass(SparqlQuery, GraphQuery)

    def test_format_for_llm_includes_query_type(self):
        """SparqlQuery format_for_llm includes SPARQL query_type."""
        q = SparqlQuery(query_string="SELECT ?s WHERE { ?s a :Person }")
        rep = q.format_for_llm()
        assert rep.args["query_type"] == "SPARQL"


# ---------------------------------------------------------------------------
# GraphResult
# ---------------------------------------------------------------------------


class TestGraphResult:
    """Tests for GraphResult Component and its format styles."""

    def test_implements_component_protocol(self, alice, matrix, acted_in):
        """GraphResult satisfies the Mellea Component protocol."""
        r = GraphResult(nodes=[alice, matrix], edges=[acted_in])
        assert isinstance(r, Component)

    def test_properties(self, alice, matrix, acted_in):
        """GraphResult exposes nodes, edges, paths, format_style."""
        r = GraphResult(nodes=[alice, matrix], edges=[acted_in], format_style="natural")
        assert r.nodes == [alice, matrix]
        assert r.edges == [acted_in]
        assert r.format_style == "natural"

    def test_empty_result(self):
        """GraphResult with no data has empty lists."""
        r = GraphResult()
        assert r.nodes == []
        assert r.edges == []
        assert r.paths == []

    def test_format_triplets(self, alice, matrix, acted_in):
        """'triplets' style formats edges as (Src)-[REL]->(Tgt)."""
        r = GraphResult(nodes=[alice, matrix], edges=[acted_in], format_style="triplets")
        rep = r.format_for_llm()
        result_text = rep.args["result"]
        assert "(Person:Alice)-[ACTED_IN]->(Movie:The Matrix)" in result_text

    def test_format_triplets_empty(self):
        """'triplets' on empty result returns a placeholder."""
        r = GraphResult(format_style="triplets")
        rep = r.format_for_llm()
        assert rep.args["result"] == "(no results)"

    def test_format_natural(self, alice, matrix, acted_in):
        """'natural' style formats edges as natural language sentences."""
        r = GraphResult(nodes=[alice, matrix], edges=[acted_in], format_style="natural")
        rep = r.format_for_llm()
        result_text = rep.args["result"]
        assert "acted in" in result_text.lower()

    def test_format_natural_empty(self):
        """'natural' on empty result returns a descriptive message."""
        r = GraphResult(format_style="natural")
        rep = r.format_for_llm()
        assert "no results" in rep.args["result"].lower()

    def test_format_paths(self, alice, matrix, acted_in, graph_path):
        """'paths' style renders paths as node-edge-node chains."""
        r = GraphResult(paths=[graph_path], format_style="paths")
        rep = r.format_for_llm()
        result_text = rep.args["result"]
        assert "ACTED_IN" in result_text
        assert "Alice" in result_text

    def test_format_paths_falls_back_to_triplets(self, alice, matrix, acted_in):
        """'paths' style falls back to triplets when there are no explicit paths."""
        r = GraphResult(nodes=[alice, matrix], edges=[acted_in], format_style="paths")
        rep = r.format_for_llm()
        assert "ACTED_IN" in rep.args["result"]

    def test_format_structured(self, alice, matrix, acted_in):
        """'structured' style returns valid JSON."""
        r = GraphResult(nodes=[alice, matrix], edges=[acted_in], format_style="structured")
        rep = r.format_for_llm()
        data = json.loads(rep.args["result"])
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

    def test_format_for_llm_includes_counts(self, alice, matrix, acted_in):
        """format_for_llm args include node_count and edge_count."""
        r = GraphResult(nodes=[alice, matrix], edges=[acted_in])
        rep = r.format_for_llm()
        assert rep.args["node_count"] == 2
        assert rep.args["edge_count"] == 1

    def test_standalone_node_appears_in_triplets(self, alice):
        """Nodes without edges appear in triplet output."""
        r = GraphResult(nodes=[alice], format_style="triplets")
        rep = r.format_for_llm()
        assert "Alice" in rep.args["result"]

    def test_parts_raises(self):
        """parts() raises NotImplementedError."""
        r = GraphResult()
        with pytest.raises(NotImplementedError):
            r.parts()


# ---------------------------------------------------------------------------
# GraphTraversal
# ---------------------------------------------------------------------------


class TestGraphTraversal:
    """Tests for GraphTraversal Component."""

    def test_implements_component_protocol(self):
        """GraphTraversal satisfies the Mellea Component protocol."""
        t = GraphTraversal(start_nodes=["Alice"])
        assert isinstance(t, Component)

    def test_properties(self):
        """GraphTraversal exposes start_nodes, pattern, max_depth, description."""
        t = GraphTraversal(
            start_nodes=["1", "2"],
            pattern="bfs",
            max_depth=4,
            description="BFS from roots",
        )
        assert t.start_nodes == ["1", "2"]
        assert t.pattern == "bfs"
        assert t.max_depth == 4
        assert t.description == "BFS from roots"

    def test_format_for_llm_returns_template_representation(self):
        """format_for_llm() returns TemplateRepresentation with a cypher field."""
        t = GraphTraversal(start_nodes=["Alice"], description="Find connections")
        rep = t.format_for_llm()
        assert isinstance(rep, TemplateRepresentation)
        assert "cypher" in rep.args
        assert rep.args["cypher"] is not None

    def test_to_cypher_multi_hop(self):
        """to_cypher() for multi_hop returns a CypherQuery."""
        t = GraphTraversal(start_nodes=["Alice"], pattern="multi_hop", max_depth=2)
        q = t.to_cypher()
        assert isinstance(q, CypherQuery)
        assert "*1..2" in q.query_string
        assert q.parameters == {"start_nodes": ["Alice"]}

    def test_to_cypher_bfs(self):
        """to_cypher() for bfs returns a CypherQuery (same pattern as multi_hop)."""
        t = GraphTraversal(start_nodes=["1"], pattern="bfs", max_depth=3)
        q = t.to_cypher()
        assert isinstance(q, CypherQuery)
        assert "*1..3" in q.query_string

    def test_to_cypher_dfs(self):
        """to_cypher() for dfs returns a CypherQuery."""
        t = GraphTraversal(start_nodes=["1"], pattern="dfs", max_depth=3)
        q = t.to_cypher()
        assert isinstance(q, CypherQuery)

    def test_to_cypher_shortest_path(self):
        """to_cypher() for shortest_path uses shortestPath()."""
        t = GraphTraversal(start_nodes=["A"], pattern="shortest_path", max_depth=5)
        q = t.to_cypher()
        assert isinstance(q, CypherQuery)
        assert "shortestPath" in q.query_string

    def test_to_cypher_unsupported_pattern(self):
        """to_cypher() raises ValueError for unknown patterns."""
        t = GraphTraversal(start_nodes=["A"], pattern="unknown_pattern")
        with pytest.raises(ValueError, match="Unsupported traversal pattern"):
            t.to_cypher()

    def test_to_cypher_preserves_description(self):
        """to_cypher() carries the traversal description into the CypherQuery."""
        t = GraphTraversal(
            start_nodes=["Alice"], pattern="multi_hop", description="My traversal"
        )
        q = t.to_cypher()
        assert q.description == "My traversal"

    def test_parts_raises(self):
        """parts() raises NotImplementedError."""
        t = GraphTraversal(start_nodes=[])
        with pytest.raises(NotImplementedError):
            t.parts()
