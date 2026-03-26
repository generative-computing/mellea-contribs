"""Graph result component (Layer 3: full Mellea Component implementation)."""

import json
from typing import TYPE_CHECKING, Any

from mellea.stdlib.components import CBlock, Component, TemplateRepresentation

from mellea_contribs.kg.base import GraphEdge, GraphNode, GraphPath

if TYPE_CHECKING:
    from mellea_contribs.kg.components.query import GraphQuery


class GraphResult(Component):
    """Component for graph query results.

    Formats query results for LLM consumption in one of several styles:
    - "triplets": ``(Subject:Label)-[EDGE_TYPE]->(Object:Label)`` format
    - "natural": Short natural language sentences per edge
    - "paths": Narrative descriptions of graph paths
    - "structured": JSON representation of nodes and edges

    Follows Mellea patterns:
    - Private fields with _ prefix
    - Public properties for read access
    - format_for_llm() returns TemplateRepresentation
    """

    def __init__(
        self,
        nodes: list[GraphNode] | None = None,
        edges: list[GraphEdge] | None = None,
        paths: list[GraphPath] | None = None,
        raw_result: Any | None = None,
        query: "GraphQuery | None" = None,
        format_style: str = "triplets",
    ):
        """Initialize a graph result.

        Args:
            nodes: Nodes returned by the query.
            edges: Edges returned by the query.
            paths: Paths returned by the query.
            raw_result: Raw backend result for debugging.
            query: The query that produced this result.
            format_style: One of "triplets", "natural", "paths", "structured".
        """
        self._nodes = nodes or []
        self._edges = edges or []
        self._paths = paths or []
        self._raw_result = raw_result
        self._query = query
        self._format_style = format_style

    # --- public properties ---

    @property
    def nodes(self) -> list[GraphNode]:
        """Nodes in the result."""
        return self._nodes

    @property
    def edges(self) -> list[GraphEdge]:
        """Edges in the result."""
        return self._edges

    @property
    def paths(self) -> list[GraphPath]:
        """Paths in the result."""
        return self._paths

    @property
    def raw_result(self) -> Any:
        """Raw backend result."""
        return self._raw_result

    @property
    def query(self) -> "GraphQuery | None":
        """The query that produced this result."""
        return self._query

    @property
    def format_style(self) -> str:
        """Active format style."""
        return self._format_style

    # --- Component protocol ---

    def parts(self) -> list[Component | CBlock]:
        """The constituent parts of this result."""
        raise NotImplementedError("parts isn't implemented by default")

    def format_for_llm(self) -> TemplateRepresentation:
        """Format the result for LLM consumption.

        Delegates to the appropriate formatter based on format_style.
        """
        formatters = {
            "triplets": self._format_triplets,
            "natural": self._format_natural,
            "paths": self._format_paths,
            "structured": self._format_structured,
        }
        formatter = formatters.get(self._format_style, self._format_triplets)
        formatted_text = formatter()

        return TemplateRepresentation(
            obj=self,
            args={
                "format_style": self._format_style,
                "result": formatted_text,
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
            },
            tools=None,
            images=None,
            template_order=["*", "GraphResult"],
        )

    # --- format helpers ---

    def _node_label(self, node: GraphNode) -> str:
        """Return a human-readable label for a node."""
        name = (
            node.properties.get("name")
            or node.properties.get("title")
            or node.properties.get("id")
            or node.id
        )
        return f"{node.label}:{name}"

    def _format_triplets(self) -> str:
        """Format as (Subject)-[PREDICATE]->(Object) triplets."""
        if not self._edges and not self._nodes:
            return "(no results)"

        lines = []
        for edge in self._edges:
            src = self._node_label(edge.source)
            tgt = self._node_label(edge.target)
            lines.append(f"({src})-[{edge.label}]->({tgt})")

        # Standalone nodes that appear in no edge
        edge_node_ids = {
            nid
            for edge in self._edges
            for nid in (edge.source.id, edge.target.id)
        }
        for node in self._nodes:
            if node.id not in edge_node_ids:
                lines.append(f"({self._node_label(node)})")

        return "\n".join(lines)

    def _format_natural(self) -> str:
        """Format as natural language sentences."""
        if not self._edges and not self._nodes:
            return "No results found."

        lines = []
        for edge in self._edges:
            src = self._node_label(edge.source)
            tgt = self._node_label(edge.target)
            rel = edge.label.replace("_", " ").lower()
            lines.append(f"{src} {rel} {tgt}.")

        edge_node_ids = {
            nid
            for edge in self._edges
            for nid in (edge.source.id, edge.target.id)
        }
        for node in self._nodes:
            if node.id not in edge_node_ids:
                lines.append(f"{self._node_label(node)} is present in the graph.")

        return "\n".join(lines)

    def _format_paths(self) -> str:
        """Format graph paths as narratives."""
        if not self._paths and not self._edges:
            return "No paths found."

        lines = []

        for path in self._paths:
            if not path.nodes:
                continue
            segments = [f"({self._node_label(path.nodes[0])})"]
            for i, edge in enumerate(path.edges):
                next_node = path.nodes[i + 1] if i + 1 < len(path.nodes) else None
                segments.append(f"-[{edge.label}]->")
                if next_node:
                    segments.append(f"({self._node_label(next_node)})")
            lines.append("".join(segments))

        # Fall back to triplets if no explicit paths
        if not lines:
            return self._format_triplets()

        return "\n".join(lines)

    def _format_structured(self) -> str:
        """Format as a JSON structure."""
        data: dict[str, Any] = {
            "nodes": [
                {"id": n.id, "label": n.label, "properties": n.properties}
                for n in self._nodes
            ],
            "edges": [
                {
                    "id": e.id,
                    "source": e.source.id,
                    "target": e.target.id,
                    "label": e.label,
                    "properties": e.properties,
                }
                for e in self._edges
            ],
        }
        return json.dumps(data, indent=2, default=str)
