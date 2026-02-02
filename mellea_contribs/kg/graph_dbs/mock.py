"""Mock backend for testing without a real graph database."""

from typing import TYPE_CHECKING, Any

from mellea_contribs.kg.base import GraphEdge, GraphNode
from mellea_contribs.kg.graph_dbs.base import GraphBackend

if TYPE_CHECKING:
    from mellea_contribs.kg.components.query import GraphQuery
    from mellea_contribs.kg.components.result import GraphResult


class MockGraphBackend(GraphBackend):
    """Mock graph backend for testing.

    Returns predefined results without connecting to a real database.
    """

    def __init__(
        self,
        mock_nodes: list[GraphNode] | None = None,
        mock_edges: list[GraphEdge] | None = None,
        mock_schema: dict[str, Any] | None = None,
        backend_options: dict | None = None,
    ):
        """Initialize mock backend.

        Args:
            mock_nodes: Predefined nodes to return
            mock_edges: Predefined edges to return
            mock_schema: Predefined schema to return
            backend_options: Additional options
        """
        super().__init__(
            backend_id="mock",
            connection_uri="mock://localhost",
            auth=None,
            database=None,
            backend_options=backend_options,
        )

        self.mock_nodes = mock_nodes or []
        self.mock_edges = mock_edges or []
        self.mock_schema = mock_schema or {
            "node_types": ["MockNode"],
            "edge_types": ["MOCK_EDGE"],
            "property_keys": ["name", "value"],
        }
        self.query_history: list[tuple[str, dict]] = []

    async def execute_query(
        self, query: "GraphQuery", **execution_options
    ) -> "GraphResult":
        """Execute a mock query.

        Records the query and returns mock results.

        Args:
            query: GraphQuery to execute
            execution_options: Additional options

        Returns:
            GraphResult with mock data
        """
        # Import here to avoid circular dependency
        from mellea_contribs.kg.components.result import GraphResult

        # Record query for testing
        self.query_history.append((query.query_string or "", query.parameters))

        # Return mock result
        return GraphResult(
            nodes=self.mock_nodes,
            edges=self.mock_edges,
            paths=[],
            raw_result=None,
            query=query,
            format_style=execution_options.get("format_style", "triplets"),
        )

    async def get_schema(self) -> dict[str, Any]:
        """Get mock schema.

        Returns:
            Mock schema dictionary
        """
        return self.mock_schema

    async def validate_query(self, query: "GraphQuery") -> tuple[bool, str | None]:
        """Validate mock query.

        Always returns True for mock queries.

        Args:
            query: GraphQuery to validate

        Returns:
            Tuple of (True, None)
        """
        return True, None

    def supports_query_type(self, query_type: str) -> bool:
        """Mock backend supports all query types.

        Args:
            query_type: Query language type

        Returns:
            True for all types
        """
        return True

    def clear_history(self):
        """Clear query history."""
        self.query_history = []
