"""Graph traversal component (Layer 2: full Mellea Component implementation)."""

from collections.abc import Callable

from mellea.stdlib.components import CBlock, Component, TemplateRepresentation

from mellea_contribs.kg.base import GraphEdge, GraphNode


class GraphTraversal(Component):
    """Component for high-level graph traversal patterns.

    Represents a multi-hop traversal intent that can be converted to a
    backend-specific query (e.g. Cypher) and formatted for LLM consumption.

    .. note::

        ``GraphTraversal`` and its ``to_cypher()`` method are **planned future
        functionality**.  The current pipeline (``kgrag.py``) builds all Cypher
        queries directly via ``@generative`` functions.  ``GraphTraversal`` will
        be integrated as a structured intermediate representation in a future
        release.

    Supported patterns:
    - "multi_hop": Follow all relationships up to max_depth hops
    - "shortest_path": Find the shortest path between two node sets
    - "bfs": Breadth-first traversal from start nodes
    - "dfs": Depth-first traversal from start nodes

    Follows Mellea patterns:
    - Private fields with _ prefix
    - Public properties for read access
    - format_for_llm() returns TemplateRepresentation
    """

    def __init__(
        self,
        start_nodes: list[str],
        pattern: str = "multi_hop",
        max_depth: int = 3,
        edge_filter: Callable[[GraphEdge], bool] | None = None,
        node_filter: Callable[[GraphNode], bool] | None = None,
        description: str | None = None,
    ):
        """Initialize a traversal pattern.

        Args:
            start_nodes: Starting node IDs or labels.
            pattern: Traversal pattern ("multi_hop", "shortest_path", "bfs", "dfs").
            max_depth: Maximum depth to traverse.
            edge_filter: Optional filter function for edges.
            node_filter: Optional filter function for nodes.
            description: Natural language description of the traversal intent.
        """
        self._start_nodes = start_nodes
        self._pattern = pattern
        self._max_depth = max_depth
        self._edge_filter = edge_filter
        self._node_filter = node_filter
        self._description = description

    # --- public properties ---

    @property
    def start_nodes(self) -> list[str]:
        """Starting node IDs or labels."""
        return self._start_nodes

    @property
    def pattern(self) -> str:
        """Traversal pattern name."""
        return self._pattern

    @property
    def max_depth(self) -> int:
        """Maximum traversal depth."""
        return self._max_depth

    @property
    def edge_filter(self) -> Callable[[GraphEdge], bool] | None:
        """Optional edge filter function."""
        return self._edge_filter

    @property
    def node_filter(self) -> Callable[[GraphNode], bool] | None:
        """Optional node filter function."""
        return self._node_filter

    @property
    def description(self) -> str | None:
        """Natural language description of the traversal."""
        return self._description

    # --- Component protocol ---

    def parts(self) -> list[Component | CBlock]:
        """The constituent parts of this traversal."""
        raise NotImplementedError("parts isn't implemented by default")

    def format_for_llm(self) -> TemplateRepresentation:
        """Format the traversal for LLM consumption."""
        return TemplateRepresentation(
            obj=self,
            args={
                "description": self._description or f"{self._pattern} traversal",
                "start_nodes": self._start_nodes,
                "pattern": self._pattern,
                "max_depth": self._max_depth,
                "cypher": self.to_cypher().query_string,
            },
            tools=None,
            images=None,
            template_order=["*", "GraphTraversal"],
        )

    # --- query conversion ---

    def to_cypher(self) -> "CypherQuery":  # type: ignore[name-defined]
        """Convert this traversal to an equivalent CypherQuery.

        Returns:
            CypherQuery that implements the traversal pattern.

        Raises:
            ValueError: If the pattern is not supported.
        """
        from mellea_contribs.kg.components.query import CypherQuery

        if self._pattern in ("multi_hop", "bfs", "dfs"):
            return (
                CypherQuery()
                .match(f"(start)-[*1..{self._max_depth}]->(end)")
                .where("start.id IN $start_nodes")
                .return_("start", "end")
                .with_parameters(start_nodes=self._start_nodes)
                .with_description(
                    self._description
                    or f"{self._pattern} traversal from {self._start_nodes}"
                )
            )

        if self._pattern == "shortest_path":
            return (
                CypherQuery()
                .match(
                    f"path = shortestPath((start)-[*1..{self._max_depth}]->(end))"
                )
                .where("start.id IN $start_nodes")
                .return_("path")
                .with_parameters(start_nodes=self._start_nodes)
                .with_description(
                    self._description
                    or f"Shortest path from {self._start_nodes}"
                )
            )

        raise ValueError(
            f"Unsupported traversal pattern: {self._pattern!r}. "
            "Supported: 'multi_hop', 'bfs', 'dfs', 'shortest_path'."
        )
