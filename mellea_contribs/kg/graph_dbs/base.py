"""Abstract backend for graph databases.

Provides a unified interface for executing graph queries across
different graph database systems (Neo4j, Neptune, RDF stores, etc.).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

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

    async def close(self):
        """Close connections to the graph database.

        Default implementation does nothing. Subclasses should override if needed.
        """
