"""Neo4j implementation of GraphBackend."""

import logging
from typing import TYPE_CHECKING, Any

_logger = logging.getLogger(__name__)

from mellea_contribs.kg.base import GraphEdge, GraphNode, GraphPath
from mellea_contribs.kg.graph_dbs.base import GraphBackend

if TYPE_CHECKING:
    from mellea_contribs.kg.components.query import GraphQuery
    from mellea_contribs.kg.components.result import GraphResult

NEO4J_AVAILABLE = False
neo4j: Any = None
GraphDatabase: Any = None
AsyncGraphDatabase: Any = None

try:
    import neo4j as _neo4j_module  # type: ignore[import-not-found]
    from neo4j import (  # type: ignore[import-not-found]
        AsyncGraphDatabase as _AsyncGraphDatabase,
        GraphDatabase as _GraphDatabase,
    )
    neo4j = _neo4j_module
    GraphDatabase = _GraphDatabase
    AsyncGraphDatabase = _AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    pass


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

    async def search_entities_by_name(self, name: str, k: int = 4) -> list[GraphNode]:
        """Search entities by case-insensitive name containment via Cypher.

        Args:
            name: Entity name fragment to search for.
            k: Maximum number of results.

        Returns:
            List of matching GraphNode objects.
        """
        cypher = (
            "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($name) "
            "RETURN n LIMIT $k"
        )
        try:
            from mellea_contribs.kg.components.query import CypherQuery
            q = CypherQuery(query_string=cypher, parameters={"name": name, "k": k})
            result = await self.execute_query(q)
            return result.nodes
        except Exception as exc:
            _logger.warning("search_entities_by_name failed: %s", exc)
            return []

    async def search_entities_by_embedding(
        self,
        embedding: list,
        k: int = 10,
        exclude_ids: set | None = None,
    ) -> list[GraphNode]:
        """Search entities using the Neo4j ``entity_embedding`` vector index.

        Falls back to an empty list when no vector index exists.

        Args:
            embedding: Query embedding vector.
            k: Maximum number of results.
            exclude_ids: Node IDs to exclude from the returned list.

        Returns:
            List of GraphNode objects ordered by similarity.
        """
        exclude_ids = exclude_ids or set()
        fetch_k = k + len(exclude_ids)
        cypher = (
            "CALL db.index.vector.queryNodes('entity_embedding', $k, $emb) "
            "YIELD node RETURN node"
        )
        try:
            from mellea_contribs.kg.components.query import CypherQuery
            q = CypherQuery(
                query_string=cypher,
                parameters={"k": fetch_k, "emb": embedding},
            )
            result = await self.execute_query(q)
            return [n for n in result.nodes if n.id not in exclude_ids][:k]
        except Exception as exc:
            _logger.warning("search_entities_by_embedding failed: %s", exc)
            return []

    async def get_relation_types(
        self, node_id: str, width: int = 30
    ) -> list[tuple[str, str]]:
        """Retrieve distinct ``(relation_type, target_label)`` pairs via Cypher.

        Args:
            node_id: Element ID of the source node.
            width: Maximum number of distinct relation types to return.

        Returns:
            List of ``(relation_type, target_label)`` tuples.
        """
        cypher = (
            "MATCH (n)-[r]->(m) WHERE elementId(n) = $nid "
            "RETURN DISTINCT type(r) AS rel_type, labels(m)[0] AS tgt_type "
            "LIMIT $w"
        )
        try:
            from mellea_contribs.kg.components.query import CypherQuery
            q = CypherQuery(
                query_string=cypher,
                parameters={"nid": node_id, "w": width},
            )
            result = await self.execute_query(q)
            pairs: list[tuple[str, str]] = []
            if result.raw_result:
                for record in result.raw_result:
                    try:
                        data = record.data() if hasattr(record, "data") else dict(record)
                        rt = data.get("rel_type")
                        tt = data.get("tgt_type") or "Unknown"
                        if rt:
                            pairs.append((str(rt), str(tt)))
                    except Exception as exc:
                        _logger.debug("Skipping malformed relation record: %s", exc)
                        continue
            if not pairs and result.edges:
                seen: set = set()
                for edge in result.edges:
                    key = (edge.label, edge.target.label)
                    if key not in seen:
                        seen.add(key)
                        pairs.append(key)
            return pairs
        except Exception as exc:
            _logger.warning("get_relation_types failed for node %s: %s", node_id, exc)
            return []

    async def get_triplets(
        self,
        node_id: str,
        rel_type: str,
        target_type: str = "Unknown",
        k: int = 30,
    ) -> list[GraphEdge]:
        """Retrieve full ``(source)-[rel]->(target)`` triplets via Cypher.

        Args:
            node_id: Element ID of the source node.
            rel_type: Relationship type to traverse.
            target_type: Label of target nodes; ignored when ``"Unknown"``/``"None"``.
            k: Maximum number of triplets to return.

        Returns:
            List of GraphEdge objects (each carries source and target GraphNodes).
        """
        # Sanitise identifiers to prevent Cypher injection.
        # Raise early if the identifier is entirely invalid rather than
        # silently producing an empty string that would generate broken Cypher.
        safe_rel = "".join(c for c in rel_type if c.isalnum() or c == "_")
        if not safe_rel:
            raise ValueError(f"Invalid relationship type for Cypher query: {rel_type!r}")
        safe_tgt = "".join(c for c in target_type if c.isalnum() or c == "_")

        if safe_tgt and safe_tgt not in ("Unknown", "None"):
            cypher = (
                f"MATCH (n)-[r:{safe_rel}]->(m:{safe_tgt}) "
                "WHERE elementId(n) = $nid RETURN n, r, m LIMIT $k"
            )
        else:
            cypher = (
                f"MATCH (n)-[r:{safe_rel}]->(m) "
                "WHERE elementId(n) = $nid RETURN n, r, m LIMIT $k"
            )
        try:
            from mellea_contribs.kg.components.query import CypherQuery
            q = CypherQuery(
                query_string=cypher,
                parameters={"nid": node_id, "k": k},
            )
            result = await self.execute_query(q)
            return result.edges
        except Exception as exc:
            _logger.warning("get_triplets failed for node %s rel %s: %s", node_id, rel_type, exc)
            return []

    async def upsert_entity(self, node: GraphNode) -> str:
        """Insert or update an entity node using ``MERGE`` on name.

        Args:
            node: GraphNode to upsert.

        Returns:
            Element ID of the created or updated node.
        """
        name = node.properties.get("name", node.id)
        props = {k: v for k, v in node.properties.items() if k != "name"}
        safe_label = "".join(c for c in node.label if c.isalnum() or c == "_") or "Entity"
        cypher = (
            f"MERGE (n:{safe_label} {{name: $name}}) "
            "SET n += $props "
            "RETURN elementId(n) AS eid"
        )
        async with self._async_driver.session(database=self.database) as session:
            result = await session.run(cypher, {"name": name, "props": props})
            record = await result.single()
            return str(record["eid"]) if record else node.id

    async def upsert_relation(self, edge: GraphEdge) -> str:
        """Insert or update a relation edge using ``MERGE`` on source/target/type.

        Args:
            edge: GraphEdge to upsert.

        Returns:
            Element ID of the created or updated edge.
        """
        safe_rel = "".join(c for c in edge.label if c.isalnum() or c == "_") or "RELATED_TO"
        props = {k: v for k, v in edge.properties.items()}
        cypher = (
            "MATCH (src) WHERE elementId(src) = $src_id "
            "MATCH (tgt) WHERE elementId(tgt) = $tgt_id "
            f"MERGE (src)-[r:{safe_rel}]->(tgt) "
            "SET r += $props "
            "RETURN elementId(r) AS eid"
        )
        async with self._async_driver.session(database=self.database) as session:
            result = await session.run(
                cypher,
                {"src_id": edge.source.id, "tgt_id": edge.target.id, "props": props},
            )
            record = await result.single()
            return str(record["eid"]) if record else edge.id

    async def fetch_entities_for_embedding(self) -> list[dict]:
        """Fetch all non-schema nodes with their element IDs and descriptions."""
        cypher = """
        MATCH (n)
        WHERE NOT n:_EntitySchema AND NOT n:_RelationSchema
        RETURN elementId(n) AS eid, n.name AS name, labels(n)[0] AS type,
               coalesce(n.description, n._description) AS description,
               n._paragraph AS paragraph
        """
        try:
            async with self._async_driver.session(database=self.database) as session:
                result = await session.run(cypher)
                records = [r async for r in result]
            return [
                {
                    "eid": r["eid"],
                    "name": r["name"] or "",
                    "type": r["type"] or "Unknown",
                    "description": " ".join(filter(None, [r["description"], r["paragraph"]])),
                }
                for r in records
            ]
        except Exception as exc:
            _logger.warning("fetch_entities_for_embedding failed: %s", exc)
            return []

    async def fetch_relations_for_embedding(self) -> list[dict]:
        """Fetch all edges with their element IDs."""
        cypher = """
        MATCH (s)-[r]->(t)
        RETURN elementId(r) AS eid, type(r) AS relation_type,
               s.name AS src_name, t.name AS dst_name
        """
        try:
            async with self._async_driver.session(database=self.database) as session:
                result = await session.run(cypher)
                records = [r async for r in result]
            return [
                {
                    "eid": r["eid"],
                    "relation_type": r["relation_type"] or "UNKNOWN",
                    "src_name": r["src_name"] or "",
                    "dst_name": r["dst_name"] or "",
                }
                for r in records
            ]
        except Exception as exc:
            _logger.warning("fetch_relations_for_embedding failed: %s", exc)
            return []

    async def store_node_embeddings(self, batch: list[dict]) -> int:
        """Store embedding vectors on nodes using the Neo4j vector property API."""
        if not batch:
            return 0
        cypher = """
        UNWIND $batch AS item
        MATCH (n) WHERE elementId(n) = item.eid
        CALL db.create.setNodeVectorProperty(n, '_embedding', item.embedding)
        RETURN count(n) AS updated
        """
        try:
            async with self._async_driver.session(database=self.database) as session:
                result = await session.run(cypher, batch=batch)
                record = await result.single()
                return record["updated"] if record else 0
        except Exception as exc:
            _logger.warning("store_node_embeddings failed: %s", exc)
            return 0

    async def store_edge_embeddings(self, batch: list[dict]) -> int:
        """Store embedding vectors on edges using the Neo4j vector property API."""
        if not batch:
            return 0
        cypher = """
        UNWIND $batch AS item
        MATCH ()-[r]->() WHERE elementId(r) = item.eid
        CALL db.create.setRelationshipVectorProperty(r, '_embedding', item.embedding)
        RETURN count(r) AS updated
        """
        try:
            async with self._async_driver.session(database=self.database) as session:
                result = await session.run(cypher, batch=batch)
                record = await result.single()
                return record["updated"] if record else 0
        except Exception as exc:
            _logger.warning("store_edge_embeddings failed: %s", exc)
            return 0

    async def create_vector_index(
        self,
        name: str,
        target: str,
        prop: str,
        dimensions: int,
        similarity: str = "cosine",
    ) -> bool:
        """Create a Neo4j vector index if it does not already exist."""
        safe_name = "".join(c for c in name if c.isalnum() or c == "_")
        safe_prop = "".join(c for c in prop if c.isalnum() or c == "_")
        safe_target = "".join(c for c in target if c.isalnum() or c == "_")
        safe_similarity = "".join(c for c in similarity if c.isalnum() or c == "_")
        if not safe_name or not safe_prop:
            raise ValueError(f"Invalid index name {name!r} or property {prop!r}")
        if target.upper() == "RELATIONSHIP":
            cypher = f"""
            CREATE VECTOR INDEX IF NOT EXISTS {safe_name}
            FOR ()-[r]-() ON (r.{safe_prop})
            OPTIONS {{indexConfig: {{`vector.dimensions`: {int(dimensions)},
                                    `vector.similarity_function`: '{safe_similarity}'}}}}
            """
        else:
            cypher = f"""
            CREATE VECTOR INDEX IF NOT EXISTS {safe_name}
            FOR (n:{safe_target}) ON (n.{safe_prop})
            OPTIONS {{indexConfig: {{`vector.dimensions`: {int(dimensions)},
                                    `vector.similarity_function`: '{safe_similarity}'}}}}
            """
        try:
            async with self._async_driver.session(database=self.database) as session:
                await session.run(cypher)
            return True
        except Exception as exc:
            _logger.debug("create_vector_index note: %s", exc)
            return False

    async def close(self):
        """Close Neo4j connections."""
        await self._async_driver.close()
        self._driver.close()
