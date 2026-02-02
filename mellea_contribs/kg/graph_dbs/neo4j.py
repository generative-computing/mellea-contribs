"""Neo4j implementation of GraphBackend."""

from typing import TYPE_CHECKING, Any

from mellea_contribs.kg.base import GraphEdge, GraphNode, GraphPath
from mellea_contribs.kg.graph_dbs.base import GraphBackend

if TYPE_CHECKING:
    from mellea_contribs.kg.components.query import GraphQuery
    from mellea_contribs.kg.components.result import GraphResult

try:
    import neo4j  # type: ignore[import-not-found]
    from neo4j import (  # type: ignore[import-not-found]
        AsyncGraphDatabase,
        GraphDatabase,
    )
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    neo4j = None  # type: ignore[assignment]
    GraphDatabase = None  # type: ignore[assignment]
    AsyncGraphDatabase = None  # type: ignore[assignment]


class Neo4jBackend(GraphBackend):
    """Neo4j implementation of GraphBackend.

    Implements the abstract GraphBackend interface for Neo4j databases.
    """

    def __init__(
        self,
        connection_uri: str = "bolt://localhost:7687",
        auth: tuple[str, str] | None = None,
        database: str | None = None,
        backend_options: dict | None = None,
    ):
        """Initialize Neo4j backend.

        Args:
            connection_uri: Neo4j connection URI
            auth: (username, password) tuple
            database: Database name (for multi-database)
            backend_options: Neo4j-specific options

        Raises:
            ImportError: If Neo4j driver is not installed
        """
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "Neo4j support requires additional dependencies. "
                "Install with: pip install mellea-contribs[kg]"
            )

        # Call parent constructor following Mellea pattern
        super().__init__(
            backend_id="neo4j",
            connection_uri=connection_uri,
            auth=auth,
            database=database,
            backend_options=backend_options,
        )

        # Create Neo4j drivers
        self._driver = GraphDatabase.driver(connection_uri, auth=auth)
        self._async_driver = AsyncGraphDatabase.driver(connection_uri, auth=auth)

    async def execute_query(
        self, query: "GraphQuery", **execution_options
    ) -> "GraphResult":
        """Execute a query in Neo4j.

        Takes a GraphQuery Component, executes it, returns GraphResult Component.

        Args:
            query: GraphQuery Component to execute
            execution_options: Additional options (format_style, etc.)

        Returns:
            GraphResult Component with parsed results

        Raises:
            ValueError: If query string is empty
        """
        # Import here to avoid circular dependency
        from mellea_contribs.kg.components.result import GraphResult

        # Get query string and parameters
        query_string = query.query_string
        parameters = query.parameters

        if not query_string:
            raise ValueError("Query string is empty")

        # Execute query
        async with self._async_driver.session(database=self.database) as session:
            result = await session.run(query_string, parameters)
            records = [record async for record in result]

        # Parse results into nodes, edges, paths
        nodes, edges, paths = self._parse_neo4j_result(records)

        # Return GraphResult Component
        return GraphResult(
            nodes=nodes,
            edges=edges,
            paths=paths,
            raw_result=records,
            query=query,
            format_style=execution_options.get("format_style", "triplets"),
        )

    def _parse_neo4j_result(
        self, records: list
    ) -> tuple[list[GraphNode], list[GraphEdge], list[GraphPath]]:
        """Parse Neo4j records into GraphNode and GraphEdge objects.

        Args:
            records: List of Neo4j records

        Returns:
            Tuple of (nodes, edges, paths)
        """
        nodes = []
        edges = []
        paths = []

        node_cache = {}  # Cache nodes by ID for edge creation
        seen_node_ids = set()
        seen_edge_ids = set()

        for record in records:
            for key in record.keys():
                value = record[key]

                if isinstance(value, neo4j.graph.Node):
                    node = GraphNode.from_neo4j_node(value)
                    if node.id not in seen_node_ids:
                        node_cache[node.id] = node
                        nodes.append(node)
                        seen_node_ids.add(node.id)

                elif isinstance(value, neo4j.graph.Relationship):
                    # Get source and target nodes
                    # Neo4j relationships always have start_node and end_node
                    assert value.start_node is not None
                    assert value.end_node is not None
                    source_id = str(value.start_node.element_id)
                    target_id = str(value.end_node.element_id)

                    # Get from cache or create
                    if source_id not in node_cache:
                        source = GraphNode.from_neo4j_node(value.start_node)
                        node_cache[source_id] = source
                        if source_id not in seen_node_ids:
                            nodes.append(source)
                            seen_node_ids.add(source_id)

                    if target_id not in node_cache:
                        target = GraphNode.from_neo4j_node(value.end_node)
                        node_cache[target_id] = target
                        if target_id not in seen_node_ids:
                            nodes.append(target)
                            seen_node_ids.add(target_id)

                    source = node_cache[source_id]
                    target = node_cache[target_id]

                    edge = GraphEdge.from_neo4j_relationship(value, source, target)
                    edge_id = str(value.element_id)
                    if edge_id not in seen_edge_ids:
                        edges.append(edge)
                        seen_edge_ids.add(edge_id)

                elif isinstance(value, neo4j.graph.Path):
                    # Parse path
                    path = GraphPath.from_neo4j_path(value)
                    paths.append(path)

                    # Also add nodes and edges to main lists if not seen
                    for node in path.nodes:
                        if node.id not in seen_node_ids:
                            nodes.append(node)
                            node_cache[node.id] = node
                            seen_node_ids.add(node.id)

                    for edge in path.edges:
                        if edge.id not in seen_edge_ids:
                            edges.append(edge)
                            seen_edge_ids.add(edge.id)

        return nodes, edges, paths

    async def get_schema(self) -> dict[str, Any]:
        """Get Neo4j schema.

        Queries for node labels, relationship types, and property keys.

        Returns:
            Dictionary with node_types, edge_types, and properties
        """
        # Get node labels
        labels_query = "CALL db.labels() YIELD label RETURN collect(label) as labels"
        async with self._async_driver.session(database=self.database) as session:
            labels_result = await session.run(labels_query)
            labels_record = await labels_result.single()
            node_types = labels_record["labels"] if labels_record else []

        # Get relationship types
        types_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"
        async with self._async_driver.session(database=self.database) as session:
            types_result = await session.run(types_query)
            types_record = await types_result.single()
            edge_types = types_record["types"] if types_record else []

        # Get property keys
        props_query = "CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) as keys"
        async with self._async_driver.session(database=self.database) as session:
            props_result = await session.run(props_query)
            props_record = await props_result.single()
            property_keys = props_record["keys"] if props_record else []

        return {
            "node_types": node_types,
            "edge_types": edge_types,
            "property_keys": property_keys,
        }

    async def validate_query(self, query: "GraphQuery") -> tuple[bool, str | None]:
        """Validate Cypher query syntax.

        Uses Neo4j's EXPLAIN to validate without executing.

        Args:
            query: GraphQuery to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            explain_query = f"EXPLAIN {query.query_string}"
            async with self._async_driver.session(database=self.database) as session:
                await session.run(explain_query, query.parameters)
            return True, None
        except neo4j.exceptions.CypherSyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Validation error: {e!s}"

    def supports_query_type(self, query_type: str) -> bool:
        """Neo4j supports Cypher queries.

        Args:
            query_type: Query language type

        Returns:
            True if "cypher", False otherwise
        """
        return query_type.lower() == "cypher"

    async def close(self):
        """Close Neo4j connections."""
        await self._async_driver.close()
        self._driver.close()
