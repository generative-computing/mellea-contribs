"""Graph result component (minimal version for Layer 4 tests)."""

from typing import TYPE_CHECKING, Any

from mellea_contribs.kg.base import GraphEdge, GraphNode, GraphPath

if TYPE_CHECKING:
    from mellea_contribs.kg.components.query import GraphQuery


class GraphResult:
    """Component for graph query results (minimal implementation for testing)."""

    def __init__(
        self,
        nodes: list[GraphNode] | None = None,
        edges: list[GraphEdge] | None = None,
        paths: list[GraphPath] | None = None,
        raw_result: Any | None = None,
        query: "GraphQuery | None" = None,
        format_style: str = "triplets",
    ):
        """Initialize graph result.

        Args:
            nodes: List of nodes in the result
            edges: List of edges in the result
            paths: List of paths (sequences of nodes/edges)
            raw_result: Raw result from backend
            query: The query that produced this result
            format_style: "triplets", "natural", "paths", "structured"
        """
        self.nodes = nodes or []
        self.edges = edges or []
        self.paths = paths or []
        self.raw_result = raw_result
        self.query = query
        self.format_style = format_style
