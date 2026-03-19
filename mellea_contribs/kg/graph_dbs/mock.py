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

    async def search_entities_by_name(self, name: str, k: int = 4) -> list[GraphNode]:
        """Return mock nodes whose name property contains the query string.

        Args:
            name: Entity name fragment to search for.
            k: Maximum number of results.

        Returns:
            Matching mock nodes (up to k).
        """
        name_lower = name.lower()
        matches = [
            n for n in self.mock_nodes
            if name_lower in str(n.properties.get("name", "")).lower()
        ]
        return matches[:k]

    async def search_entities_by_embedding(
        self,
        embedding: list,
        k: int = 10,
        exclude_ids: set | None = None,
    ) -> list[GraphNode]:
        """Return mock nodes excluding excluded IDs (no real vector search).

        Args:
            embedding: Query embedding vector (unused in mock).
            k: Maximum number of results.
            exclude_ids: Node IDs to exclude.

        Returns:
            Mock nodes up to k, excluding any in exclude_ids.
        """
        exclude_ids = exclude_ids or set()
        results = [n for n in self.mock_nodes if n.id not in exclude_ids]
        return results[:k]

    async def get_relation_types(
        self, node_id: str, width: int = 30
    ) -> list[tuple[str, str]]:
        """Return distinct (relation_type, target_label) pairs from mock edges.

        Args:
            node_id: Source node element ID.
            width: Maximum number of pairs.

        Returns:
            List of (relation_type, target_label) tuples.
        """
        pairs: list[tuple[str, str]] = []
        seen: set = set()
        for edge in self.mock_edges:
            if edge.source.id == node_id:
                key = (edge.label, edge.target.label)
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
        return pairs[:width]

    async def get_triplets(
        self,
        node_id: str,
        rel_type: str,
        target_type: str = "Unknown",
        k: int = 30,
    ) -> list[GraphEdge]:
        """Return mock edges matching source node and relation type.

        Args:
            node_id: Source node element ID.
            rel_type: Relationship type to match.
            target_type: Target label filter (ignored when "Unknown"/"None").
            k: Maximum number of triplets.

        Returns:
            Matching GraphEdge objects.
        """
        matches = [
            e for e in self.mock_edges
            if e.source.id == node_id and e.label == rel_type
            and (
                target_type in ("Unknown", "None", "")
                or e.target.label == target_type
            )
        ]
        return matches[:k]

    async def upsert_entity(self, node: GraphNode) -> str:
        """Insert or update an entity node in the mock store.

        Args:
            node: GraphNode to upsert.

        Returns:
            The node's id.
        """
        for i, existing in enumerate(self.mock_nodes):
            if existing.id == node.id:
                self.mock_nodes[i] = node
                return node.id
        self.mock_nodes.append(node)
        return node.id

    async def upsert_relation(self, edge: GraphEdge) -> str:
        """Insert or update a relation edge in the mock store.

        Args:
            edge: GraphEdge to upsert.

        Returns:
            The edge's id.
        """
        for i, existing in enumerate(self.mock_edges):
            if existing.id == edge.id:
                self.mock_edges[i] = edge
                return edge.id
        self.mock_edges.append(edge)
        return edge.id
