"""Tests for Layer 3: LLM-guided query construction.

Covers:
- components/llm_guided.py: @generative functions and GeneratedQuery model
- sampling/validation.py: QueryValidationStrategy
- requirements/__init__.py: is_valid_cypher, returns_results, respects_schema
"""

import pytest
from mellea.stdlib.components import CBlock, ModelOutputThunk
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.requirements import Requirement, ValidationResult

from mellea_contribs.kg.base import GraphEdge, GraphNode
from mellea_contribs.kg.components.llm_guided import (
    GeneratedQuery,
    explain_query_result,
    natural_language_to_cypher,
    suggest_query_improvement,
)
from mellea_contribs.kg.graph_dbs.mock import MockGraphBackend
from mellea_contribs.kg.requirements import (
    is_valid_cypher,
    respects_schema,
    returns_results,
)
from mellea_contribs.kg.sampling import QueryValidationStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_backend():
    """Create a mock backend with sample data."""
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


@pytest.fixture
def empty_backend():
    """Create a mock backend with no data."""
    return MockGraphBackend()


@pytest.fixture
def ctx_with_query():
    """Create a SimpleContext whose last output is a Cypher query string."""
    ctx = SimpleContext()
    thunk = ModelOutputThunk(value="MATCH (n:Person) RETURN n")
    return SimpleContext.from_previous(ctx, thunk)


# ---------------------------------------------------------------------------
# GeneratedQuery model
# ---------------------------------------------------------------------------


class TestGeneratedQuery:
    """Tests for the GeneratedQuery Pydantic model."""

    def test_create_with_all_fields(self):
        """GeneratedQuery can be created with all fields."""
        gq = GeneratedQuery(
            query="MATCH (n) RETURN n",
            explanation="Returns all nodes",
            parameters={"limit": 10},
        )
        assert gq.query == "MATCH (n) RETURN n"
        assert gq.explanation == "Returns all nodes"
        assert gq.parameters == {"limit": 10}

    def test_parameters_optional(self):
        """GeneratedQuery parameters field defaults to None."""
        gq = GeneratedQuery(query="MATCH (n) RETURN n", explanation="All nodes")
        assert gq.parameters is None

    def test_is_pydantic_model(self):
        """GeneratedQuery is a Pydantic BaseModel."""
        from pydantic import BaseModel

        assert issubclass(GeneratedQuery, BaseModel)


# ---------------------------------------------------------------------------
# @generative functions
# ---------------------------------------------------------------------------


class TestGenerativeFunctions:
    """Tests for the @generative LLM-guided functions."""

    def test_natural_language_to_cypher_is_callable(self):
        """natural_language_to_cypher is a callable generative slot."""
        assert callable(natural_language_to_cypher)

    def test_explain_query_result_is_callable(self):
        """explain_query_result is a callable generative slot."""
        assert callable(explain_query_result)

    def test_suggest_query_improvement_is_callable(self):
        """suggest_query_improvement is a callable generative slot."""
        assert callable(suggest_query_improvement)

    def test_functions_have_correct_names(self):
        """@generative functions preserve their original names."""
        assert natural_language_to_cypher.__name__ == "natural_language_to_cypher"
        assert explain_query_result.__name__ == "explain_query_result"
        assert suggest_query_improvement.__name__ == "suggest_query_improvement"


# ---------------------------------------------------------------------------
# QueryValidationStrategy
# ---------------------------------------------------------------------------


class TestQueryValidationStrategy:
    """Tests for QueryValidationStrategy."""

    def test_create_strategy(self, mock_backend):
        """QueryValidationStrategy can be created with a backend."""
        strategy = QueryValidationStrategy(backend=mock_backend)
        assert strategy._backend is mock_backend
        assert strategy.loop_budget == 3

    def test_create_strategy_custom_budget(self, mock_backend):
        """QueryValidationStrategy respects a custom loop_budget."""
        strategy = QueryValidationStrategy(backend=mock_backend, loop_budget=5)
        assert strategy.loop_budget == 5

    def test_create_strategy_with_requirements(self, mock_backend):
        """QueryValidationStrategy stores requirements."""
        req = is_valid_cypher(mock_backend)
        strategy = QueryValidationStrategy(
            backend=mock_backend, requirements=[req]
        )
        assert strategy.requirements == [req]

    def test_repair_returns_cblock_and_context(self, ctx_with_query):
        """repair() returns a (CBlock, Context) tuple."""
        failed_thunk = ModelOutputThunk(value="MATCH (n) RETURN n WRONG")
        val_result = ValidationResult(False, reason="Syntax error near WRONG")
        from mellea.stdlib.requirements import Requirement

        req = Requirement(description="valid cypher")

        action, new_ctx = QueryValidationStrategy.repair(
            old_ctx=ctx_with_query,
            new_ctx=ctx_with_query,
            past_actions=[],
            past_results=[failed_thunk],
            past_val=[[(req, val_result)]],
        )

        assert isinstance(action, CBlock)
        assert "Syntax error near WRONG" in str(action)
        assert "MATCH (n) RETURN n WRONG" in str(action)

    def test_repair_handles_multiple_errors(self, ctx_with_query):
        """repair() collects all error messages from the last validation."""
        thunk = ModelOutputThunk(value="bad query")
        req1 = Requirement(description="req1")
        req2 = Requirement(description="req2")
        val = [
            (req1, ValidationResult(False, reason="Error A")),
            (req2, ValidationResult(False, reason="Error B")),
        ]

        action, _ = QueryValidationStrategy.repair(
            old_ctx=ctx_with_query,
            new_ctx=ctx_with_query,
            past_actions=[],
            past_results=[thunk],
            past_val=[val],
        )

        assert "Error A" in str(action)
        assert "Error B" in str(action)

    def test_select_from_failure_picks_fewest_errors(self):
        """select_from_failure returns the index with the fewest failures."""
        req = Requirement(description="req")
        val_0 = [(req, ValidationResult(False)), (req, ValidationResult(False))]
        val_1 = [(req, ValidationResult(False))]
        val_2 = [(req, ValidationResult(True))]

        idx = QueryValidationStrategy.select_from_failure(
            sampled_actions=[],
            sampled_results=[],
            sampled_val=[val_0, val_1, val_2],
        )

        assert idx == 2

    def test_select_from_failure_tie_picks_first(self):
        """select_from_failure picks the first index when error counts tie."""
        req = Requirement(description="req")
        val = [(req, ValidationResult(False))]

        idx = QueryValidationStrategy.select_from_failure(
            sampled_actions=[],
            sampled_results=[],
            sampled_val=[val, val],
        )

        assert idx == 0


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------


class TestRequirements:
    """Tests for graph-specific Requirement factories."""

    def test_is_valid_cypher_returns_requirement(self, mock_backend):
        """is_valid_cypher() returns a Requirement."""
        req = is_valid_cypher(mock_backend)
        assert isinstance(req, Requirement)
        assert "Cypher" in req.description

    def test_returns_results_returns_requirement(self, mock_backend):
        """returns_results() returns a Requirement."""
        req = returns_results(mock_backend)
        assert isinstance(req, Requirement)
        assert "results" in req.description.lower()

    def test_respects_schema_returns_requirement(self, mock_backend):
        """respects_schema() returns a Requirement."""
        req = respects_schema(mock_backend)
        assert isinstance(req, Requirement)
        assert "schema" in req.description.lower()

    def test_all_requirements_have_validation_fn(self, mock_backend):
        """All requirement factories set a validation_fn."""
        for req in [
            is_valid_cypher(mock_backend),
            returns_results(mock_backend),
            respects_schema(mock_backend),
        ]:
            assert req.validation_fn is not None

    async def test_is_valid_cypher_passes_for_valid_query(
        self, mock_backend, ctx_with_query
    ):
        """is_valid_cypher validation passes when backend validates OK."""
        req = is_valid_cypher(mock_backend)
        result = await req.validation_fn(ctx_with_query)
        assert isinstance(result, ValidationResult)
        assert bool(result) is True

    async def test_returns_results_passes_when_data_present(
        self, mock_backend, ctx_with_query
    ):
        """returns_results validation passes when backend has nodes/edges."""
        req = returns_results(mock_backend)
        result = await req.validation_fn(ctx_with_query)
        assert isinstance(result, ValidationResult)
        assert bool(result) is True

    async def test_returns_results_fails_when_no_data(
        self, empty_backend, ctx_with_query
    ):
        """returns_results validation fails when backend has no data."""
        req = returns_results(empty_backend)
        result = await req.validation_fn(ctx_with_query)
        assert isinstance(result, ValidationResult)
        assert bool(result) is False
        assert "no results" in result.reason.lower()

    async def test_respects_schema_passes(self, mock_backend, ctx_with_query):
        """respects_schema validation passes (placeholder implementation)."""
        req = respects_schema(mock_backend)
        result = await req.validation_fn(ctx_with_query)
        assert isinstance(result, ValidationResult)
        assert bool(result) is True
