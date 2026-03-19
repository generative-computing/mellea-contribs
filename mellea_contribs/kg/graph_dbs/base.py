"""Abstract backend for graph databases.

Provides a unified interface for executing graph queries across
different graph database systems (Neo4j, Neptune, RDF stores, etc.).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from mellea_contribs.kg.base import GraphEdge, GraphNode

if TYPE_CHECKING:
    from mellea_contribs.kg.components.query import GraphQuery
    from mellea_contribs.kg.components.result import GraphResult
    from mellea_contribs.kg.components.traversal import GraphTraversal


class GraphBackend(ABC):
    """Abstract backend for graph databases.

    Following Mellea's Backend pattern:
    - Takes backend_id (like model_id)
    - Takes backend_options (like model_options)
    - Abstract methods for core operations
    """

    def __init__(
        self,
        backend_id: str,
        *,
        connection_uri: str | None = None,
        auth: tuple[str, str] | None = None,
        database: str | None = None,
        backend_options: dict | None = None,
    ):
        """Initialize graph backend.

        Following Mellea's Backend(model_id, model_options) pattern.

        Args:
            backend_id: Identifier for backend type (e.g., "neo4j", "neptune")
            connection_uri: URI for connecting to the database
            auth: (username, password) tuple for authentication
            database: Database name (if multi-database system)
            backend_options: Backend-specific options
        """
        # MELLEA PATTERN: Similar to Backend.__init__
        self.backend_id = backend_id
        self.backend_options = backend_options if backend_options is not None else {}

        # Graph-specific fields
        self.connection_uri = connection_uri
        self.auth = auth
        self.database = database

    @abstractmethod
    async def execute_query(
        self, query: "GraphQuery", **execution_options
    ) -> "GraphResult":
        """Execute a graph query and return results.

        Similar to Backend.generate_from_context() for LLMs.
        Takes a Component (GraphQuery), returns a Component (GraphResult).

        Args:
            query: The GraphQuery Component to execute
            execution_options: Backend-specific execution options

        Returns:
            GraphResult Component containing formatted results
        """
        ...

    @abstractmethod
    async def get_schema(self) -> dict[str, Any]:
        """Get the graph schema.

        Returns:
            Dictionary with node_types, edge_types, properties, etc.
        """
        ...

    @abstractmethod
    async def validate_query(self, query: "GraphQuery") -> tuple[bool, str | None]:
        """Validate query syntax and semantics.

        Args:
            query: The GraphQuery to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        ...

    def supports_query_type(self, query_type: str) -> bool:
        """Check if this backend supports a query type (Cypher, SPARQL, etc.).

        Default implementation returns False. Subclasses should override.

        Args:
            query_type: Query language type (e.g., "cypher", "sparql")

        Returns:
            True if supported, False otherwise
        """
        return False

    async def execute_traversal(
        self, traversal: "GraphTraversal", **execution_options
    ) -> "GraphResult":
        """Execute a high-level traversal pattern.

        Default implementation converts to backend-specific query.

        Args:
            traversal: The GraphTraversal pattern to execute
            execution_options: Backend-specific execution options

        Returns:
            GraphResult Component containing formatted results
        """
        if self.supports_query_type("cypher"):
            query = traversal.to_cypher()
            return await self.execute_query(query, **execution_options)
        else:
            raise NotImplementedError(
                f"Traversal not implemented for {self.__class__.__name__}"
            )

    @abstractmethod
    async def search_entities_by_name(self, name: str, k: int = 4) -> list[GraphNode]:
        """Search entities by case-insensitive name containment.

        Args:
            name: Entity name fragment to search for.
            k: Maximum number of results.

        Returns:
            List of matching GraphNode objects.
        """
        ...

    @abstractmethod
    async def search_entities_by_embedding(
        self,
        embedding: list,
        k: int = 10,
        exclude_ids: set | None = None,
    ) -> list[GraphNode]:
        """Search entities using a vector similarity index.

        Falls back gracefully to an empty list when no vector index exists.

        Args:
            embedding: Query embedding vector.
            k: Maximum number of results.
            exclude_ids: Node IDs to exclude from the returned list.

        Returns:
            List of GraphNode objects ordered by similarity.
        """
        ...

    @abstractmethod
    async def get_relation_types(
        self, node_id: str, width: int = 30
    ) -> list[tuple[str, str]]:
        """Retrieve distinct ``(relation_type, target_label)`` pairs from a node.

        Args:
            node_id: Element ID of the source node.
            width: Maximum number of distinct relation types to return.

        Returns:
            List of ``(relation_type, target_label)`` tuples.
        """
        ...

    @abstractmethod
    async def get_triplets(
        self,
        node_id: str,
        rel_type: str,
        target_type: str = "Unknown",
        k: int = 30,
    ) -> list[GraphEdge]:
        """Retrieve full ``(source)-[rel]->(target)`` triplets from the graph.

        Args:
            node_id: Element ID of the source node.
            rel_type: Relationship type to traverse.
            target_type: Label of target nodes; ignored when ``"Unknown"``/``"None"``.
            k: Maximum number of triplets to return.

        Returns:
            List of GraphEdge objects (each carries source and target GraphNodes).
        """
        ...

    @abstractmethod
    async def upsert_entity(self, node: GraphNode) -> str:
        """Insert or update an entity node in the graph.

        Uses merge semantics: creates the node if it does not exist, updates
        properties if it does.

        Args:
            node: GraphNode to upsert.  ``node.label`` is used as the node
                label and ``node.properties["name"]`` as the merge key when
                present.

        Returns:
            The element ID of the created or updated node.
        """
        ...

    @abstractmethod
    async def upsert_relation(self, edge: GraphEdge) -> str:
        """Insert or update a relation edge in the graph.

        Uses merge semantics based on source node, target node, and relation
        label.

        Args:
            edge: GraphEdge to upsert.  ``edge.source`` and ``edge.target``
                must have valid element IDs already present in the graph.

        Returns:
            The element ID of the created or updated edge.
        """
        ...

    async def close(self):
        """Close connections to the graph database.

        Default implementation does nothing. Subclasses should override if needed.
        """
