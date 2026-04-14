"""Graph query components (Layer 3: full Mellea Component implementations)."""

from copy import deepcopy
from typing import Any

from mellea.stdlib.components import CBlock, Component, TemplateRepresentation, blockify


class GraphQuery(Component):
    """Base Component for graph queries.

    Represents a graph query that can be executed against a GraphBackend
    and formatted for LLM consumption.

    """

    def __init__(
        self,
        query_string: str | CBlock | None = None,
        parameters: dict | None = None,
        description: str | CBlock | None = None,
        metadata: dict | None = None,
    ):
        """Initialize a graph query.

        Args:
            query_string: The actual query (Cypher, SPARQL, etc.)
            parameters: Query parameters for parameterized queries
            description: Natural language description of what the query does
            metadata: Additional metadata (schema hints, temporal constraints, etc.)
        """
        self._query_string = blockify(query_string) if query_string is not None else None
        self._parameters = parameters or {}
        self._description = blockify(description) if description is not None else None
        self._metadata = metadata or {}

    @property
    def query_string(self) -> str | None:
        """The query string."""
        return str(self._query_string) if self._query_string is not None else None

    @property
    def parameters(self) -> dict[str, Any]:
        """Query parameters."""
        return self._parameters

    @property
    def description(self) -> str | None:
        """Natural language description of the query."""
        return str(self._description) if self._description is not None else None

    @property
    def metadata(self) -> dict[str, Any]:
        """Additional query metadata."""
        return self._metadata

    def parts(self) -> list[Component | CBlock]:
        """The constituent parts of this query."""
        raise NotImplementedError("parts isn't implemented by default")

    def format_for_llm(self) -> TemplateRepresentation:
        """Format query for LLM consumption."""
        return TemplateRepresentation(
            obj=self,
            args={
                "description": self.description or "Graph query",
                "query": self.query_string,
                "parameters": self._parameters,
                "metadata": self._metadata,
            },
            tools=None,
            images=None,
            template_order=["*", "GraphQuery"],
        )

    def with_description(self, description: str | CBlock) -> "GraphQuery":
        """Return a new query with an updated description (immutable).

        Args:
            description: New natural language description.

        Returns:
            New GraphQuery with the updated description.
        """
        result = deepcopy(self)
        result._description = blockify(description)
        return result

    def with_parameters(self, **params: Any) -> "GraphQuery":
        """Return a new query with additional parameters merged in (immutable).

        Args:
            **params: Parameters to merge into the current set.

        Returns:
            New GraphQuery with the merged parameters.
        """
        result = deepcopy(self)
        result._parameters = {**self._parameters, **params}
        return result

    def with_metadata(self, **metadata: Any) -> "GraphQuery":
        """Return a new query with additional metadata merged in (immutable).

        Args:
            **metadata: Metadata entries to merge.

        Returns:
            New GraphQuery with the merged metadata.
        """
        result = deepcopy(self)
        result._metadata = {**self._metadata, **metadata}
        return result


class CypherQuery(GraphQuery):
    """Component for building Cypher queries (Neo4j).

    Provides a fluent, composable interface for building Cypher queries.
    Each builder method returns a new instance (immutable).

    Example:
        query = (
            CypherQuery()
            .match("(m:Movie)")
            .where("m.year = $year")
            .return_("m.title", "m.year")
            .order_by("m.year DESC")
            .limit(10)
            .with_parameters(year=2020)
        )
    """

    def __init__(
        self,
        query_string: str | CBlock | None = None,
        parameters: dict | None = None,
        description: str | CBlock | None = None,
        metadata: dict | None = None,
        match_clauses: list[str] | None = None,
        where_clauses: list[str] | None = None,
        return_clauses: list[str] | None = None,
        order_by: list[str] | None = None,
        limit: int | None = None,
    ):
        """Initialize a Cypher query builder.

        Args:
            query_string: Explicit query string (bypasses clause building when provided).
            parameters: Query parameters.
            description: Natural language description.
            metadata: Additional metadata.
            match_clauses: MATCH clause patterns.
            where_clauses: WHERE conditions.
            return_clauses: RETURN expressions.
            order_by: ORDER BY expressions.
            limit: LIMIT value.
        """
        self._match_clauses: list[str] = match_clauses or []
        self._where_clauses: list[str] = where_clauses or []
        self._return_clauses: list[str] = return_clauses or []
        self._order_by: list[str] = order_by or []
        self._limit: int | None = limit

        # Build query string from clauses if not provided
        if query_string is None and self._match_clauses:
            query_string = self._build_query_string(
                self._match_clauses,
                self._where_clauses,
                self._return_clauses,
                self._order_by,
                self._limit,
            )

        super().__init__(query_string, parameters, description, metadata)

    @staticmethod
    def _build_query_string(
        match: list[str],
        where: list[str],
        return_: list[str],
        order: list[str],
        limit: int | None,
    ) -> str:
        """Build a Cypher query string from clause lists."""
        parts = []
        if match:
            parts.append("MATCH " + ", ".join(match))
        if where:
            parts.append("WHERE " + " AND ".join(where))
        if return_:
            parts.append("RETURN " + ", ".join(return_))
        if order:
            parts.append("ORDER BY " + ", ".join(order))
        if limit is not None:
            parts.append(f"LIMIT {limit}")
        return "\n".join(parts)

    def _rebuild(self) -> "CypherQuery":
        """Rebuild the query string from the current clause lists."""
        if self._match_clauses or self._return_clauses:
            self._query_string = blockify(
                self._build_query_string(
                    self._match_clauses,
                    self._where_clauses,
                    self._return_clauses,
                    self._order_by,
                    self._limit,
                )
            )
        return self

    def match(self, pattern: str) -> "CypherQuery":
        """Add a MATCH clause (immutable).

        Args:
            pattern: Cypher MATCH pattern, e.g. "(n:Person)".

        Returns:
            New CypherQuery with the clause appended.
        """
        result = deepcopy(self)
        result._match_clauses = [*self._match_clauses, pattern]
        return result._rebuild()

    def where(self, condition: str) -> "CypherQuery":
        """Add a WHERE condition (immutable).

        Args:
            condition: Cypher WHERE condition, e.g. "n.age > 30".

        Returns:
            New CypherQuery with the condition appended.
        """
        result = deepcopy(self)
        result._where_clauses = [*self._where_clauses, condition]
        return result._rebuild()

    def return_(self, *items: str) -> "CypherQuery":
        """Add RETURN expressions (immutable).

        Args:
            *items: One or more RETURN expressions.

        Returns:
            New CypherQuery with the expressions appended.
        """
        result = deepcopy(self)
        result._return_clauses = [*self._return_clauses, *items]
        return result._rebuild()

    def order_by(self, *fields: str) -> "CypherQuery":
        """Add ORDER BY fields (immutable).

        Args:
            *fields: One or more ORDER BY expressions.

        Returns:
            New CypherQuery with the fields appended.
        """
        result = deepcopy(self)
        result._order_by = [*self._order_by, *fields]
        return result._rebuild()

    def limit(self, n: int) -> "CypherQuery":
        """Set the LIMIT clause (immutable).

        Args:
            n: Maximum number of results.

        Returns:
            New CypherQuery with the limit set.
        """
        result = deepcopy(self)
        result._limit = n
        return result._rebuild()

    def format_for_llm(self) -> TemplateRepresentation:
        """Format the Cypher query for LLM consumption."""
        return TemplateRepresentation(
            obj=self,
            args={
                "description": self.description or "Cypher graph query",
                "query": self.query_string,
                "parameters": self._parameters,
                "query_type": "Cypher (Neo4j)",
            },
            tools=None,
            images=None,
            template_order=["*", "CypherQuery", "GraphQuery"],
        )


class SparqlQuery(GraphQuery):
    """Component for SPARQL queries (RDF/triple stores).

    Extends GraphQuery with SPARQL-specific formatting.

    .. note::

        ``SparqlQuery`` is **planned future functionality** for RDF/triple-store
        backends.  The current production backend is Neo4j (``CypherQuery``).
        ``SparqlQuery`` is exported for completeness but has no callers in the
        current pipeline.
    """

    def format_for_llm(self) -> TemplateRepresentation:
        """Format the SPARQL query for LLM consumption."""
        return TemplateRepresentation(
            obj=self,
            args={
                "description": self.description or "SPARQL graph query",
                "query": self.query_string,
                "parameters": self._parameters,
                "query_type": "SPARQL",
            },
            tools=None,
            images=None,
            template_order=["*", "SparqlQuery", "GraphQuery"],
        )
