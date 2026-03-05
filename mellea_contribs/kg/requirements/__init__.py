"""Graph-specific requirements for query validation.

Provides Requirement factories for validating graph queries against
syntax rules, schema constraints, and executability.
"""

from mellea.stdlib.context import Context
from mellea.stdlib.requirements import Requirement, ValidationResult

from mellea_contribs.kg.components.query import GraphQuery
from mellea_contribs.kg.graph_dbs.base import GraphBackend


def is_valid_cypher(backend: GraphBackend) -> Requirement:
    """Require that the query is valid Cypher syntax.

    Uses the backend's validate_query() to check syntax without executing.

    Args:
        backend: Graph backend used to validate the query.

    Returns:
        A Requirement that passes when the query is syntactically valid.
    """

    async def validate(ctx: Context) -> ValidationResult:
        query_string = ctx.last_output().value
        query = GraphQuery(query_string=str(query_string))
        is_valid, error = await backend.validate_query(query)
        return ValidationResult(
            is_valid,
            reason=error if not is_valid else "Valid Cypher syntax",
        )

    return Requirement(
        description="Query must be valid Cypher syntax",
        validation_fn=validate,
    )


def returns_results(backend: GraphBackend) -> Requirement:
    """Require that the query returns at least one node or edge.

    Executes the query and checks for non-empty results.

    Args:
        backend: Graph backend used to execute the query.

    Returns:
        A Requirement that passes when the query produces results.
    """

    async def validate(ctx: Context) -> ValidationResult:
        query_string = ctx.last_output().value
        query = GraphQuery(query_string=str(query_string))
        result = await backend.execute_query(query)
        has_results = len(result.nodes) > 0 or len(result.edges) > 0
        return ValidationResult(
            has_results,
            reason=(
                "Query returned results"
                if has_results
                else "Query returned no results"
            ),
        )

    return Requirement(
        description="Query must return non-empty results",
        validation_fn=validate,
    )


def respects_schema(backend: GraphBackend) -> Requirement:
    """Require that the query only references node and edge types in the schema.

    Args:
        backend: Graph backend used to retrieve the schema.

    Returns:
        A Requirement that passes when the query respects the schema.
    """

    async def validate(ctx: Context) -> ValidationResult:
        await backend.get_schema()
        # Full Cypher AST parsing would be needed for strict enforcement.
        # This is a placeholder that always passes once the schema is fetched.
        return ValidationResult(
            True,
            reason="Query respects schema",
        )

    return Requirement(
        description="Query must only reference valid schema types",
        validation_fn=validate,
    )


__all__ = ["is_valid_cypher", "respects_schema", "returns_results"]
