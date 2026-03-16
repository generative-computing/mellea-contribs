"""KGRag: Knowledge Graph Retrieval-Augmented Generation.

Layer 1 application that orchestrates the full KG RAG pipeline:

**QA Pipeline**:
  1. Break down question into solving routes (Layer 3 @generative)
  2. Extract topic entities from routes (Layer 3 @generative)
  3. Align entities with KG candidates (Layer 3 @generative)
  4. Prune relevant relations (Layer 3 @generative)
  5. Evaluate knowledge sufficiency (Layer 3 @generative)
  6. Generate answer or validate consensus (Layer 3 @generative)

**Update Pipeline**:
  1. Extract entities and relations from document (Layer 3 @generative)
  2. Align extracted entities with KG (Layer 3 @generative)
  3. Decide entity merges (Layer 3 @generative)
  4. Align extracted relations with KG (Layer 3 @generative)
  5. Decide relation merges (Layer 3 @generative)

Example::

    import asyncio
    from mellea import start_session
    from mellea_contribs.kg import Neo4jBackend, KGRag

    async def main():
        session = start_session(backend_name="litellm", model_id="gpt-4o-mini")
        backend = Neo4jBackend(
            connection_uri="bolt://localhost:7687",
            auth=("neo4j", "password"),
        )
        rag = KGRag(backend=backend, session=session)
        answer = await rag.answer("Who acted in The Matrix?")
        print(answer)
        await backend.close()

    asyncio.run(main())
"""

from typing import Any, Optional

try:
    from mellea import MelleaSession
except ImportError:
    MelleaSession = None  # type: ignore[assignment,misc]

# Optional imports from mellea components (requires mellea to be installed)
try:
    from mellea_contribs.kg.components import (
        align_entity_with_kg,
        align_relation_with_kg,
        align_topic_entities,
        break_down_question,
        decide_entity_merge,
        decide_relation_merge,
        evaluate_knowledge_sufficiency,
        extract_entities_and_relations,
        extract_topic_entities,
        generate_direct_answer,
        prune_relations,
        prune_triplets,
        validate_consensus,
    )
    from mellea_contribs.kg.components.llm_guided import (
        explain_query_result,
        natural_language_to_cypher,
        suggest_query_improvement,
    )
    from mellea_contribs.kg.components.query import CypherQuery
    from mellea_contribs.kg.components.result import GraphResult
except ImportError:
    # These are optional - mellea may not be installed
    align_entity_with_kg = None
    align_relation_with_kg = None
    align_topic_entities = None
    break_down_question = None
    decide_entity_merge = None
    decide_relation_merge = None
    evaluate_knowledge_sufficiency = None
    extract_entities_and_relations = None
    extract_topic_entities = None
    generate_direct_answer = None
    prune_relations = None
    prune_triplets = None
    validate_consensus = None
    explain_query_result = None
    natural_language_to_cypher = None
    suggest_query_improvement = None
    CypherQuery = None
    GraphResult = None

from mellea_contribs.kg.graph_dbs.base import GraphBackend

# Maximum Cypher repair attempts before giving up
_MAX_REPAIR_ATTEMPTS = 2


def format_schema(schema: dict) -> str:
    """Format a graph schema dictionary into a readable string for LLM prompts.

    Args:
        schema: Dictionary with "node_types", "edge_types", and "property_keys"
                keys (as returned by ``GraphBackend.get_schema()``).

    Returns:
        A human-readable schema description.
    """
    node_types = schema.get("node_types", [])
    edge_types = schema.get("edge_types", [])
    property_keys = schema.get("property_keys", [])

    lines = ["Graph Schema:"]
    if node_types:
        lines.append(f"  Node labels: {', '.join(node_types)}")
    if edge_types:
        lines.append(f"  Relationship types: {', '.join(edge_types)}")
    if property_keys:
        lines.append(f"  Property keys: {', '.join(property_keys)}")

    return "\n".join(lines)


class KGRag:
    """Knowledge Graph Retrieval-Augmented Generation pipeline.

    Combines a Mellea session (for LLM calls) with a graph backend (for query
    execution) to answer natural language questions about a knowledge graph.

    The pipeline for each question:

    1. **Schema retrieval** — fetch the current graph schema so the LLM knows
       what node labels and relationship types exist.
    2. **Query generation** — ``natural_language_to_cypher`` converts the
       question into a Cypher query via a ``@generative`` LLM call.
    3. **Validation & repair** — the generated Cypher is validated against the
       database.  If invalid, ``suggest_query_improvement`` is called (up to
       ``max_repair_attempts`` times) to produce a corrected query.
    4. **Execution** — the validated query is executed against the backend.
    5. **Answer generation** — ``explain_query_result`` produces a natural
       language answer grounded in the query results.

    Args:
        backend: Graph database backend (Layer 4).
        session: Active Mellea session wrapping an LLM backend.
        format_style: How query results are formatted for the LLM
            ("triplets", "natural", "paths", or "structured").
        max_repair_attempts: Maximum number of Cypher repair attempts before
            the pipeline gives up and returns whatever was last generated.
    """

    def __init__(
        self,
        backend: GraphBackend,
        session: MelleaSession,
        format_style: str = "natural",
        max_repair_attempts: int = _MAX_REPAIR_ATTEMPTS,
    ):
        """Initialize a KGRag pipeline.

        Args:
            backend: Graph database backend.
            session: Mellea session for LLM calls.
            format_style: Result format style passed to GraphResult.
            max_repair_attempts: Max Cypher repair attempts.
        """
        self._backend = backend
        self._session = session
        self._format_style = format_style
        self._max_repair_attempts = max_repair_attempts

    async def answer(self, question: str, examples: str = "") -> str:
        """Answer a natural language question using the knowledge graph.

        Args:
            question: A natural language question about the graph data.
            examples: Optional few-shot Cypher examples to guide generation.

        Returns:
            A natural language answer grounded in graph query results.
        """
        # Step 1: Get graph schema
        schema = await self._backend.get_schema()
        schema_text = format_schema(schema)

        # Step 2: Generate Cypher query from natural language
        generated = await natural_language_to_cypher(
            self._session,
            natural_language_query=question,
            graph_schema=schema_text,
            examples=examples,
        )
        cypher_string = generated.query

        # Step 3: Validate and repair loop
        cypher_string = await self._validate_and_repair(
            cypher_string, schema_text
        )

        # Step 4: Execute validated query
        query = CypherQuery(query_string=cypher_string, description=question)
        graph_result = await self._backend.execute_query(
            query, format_style=self._format_style
        )

        # Step 5: Generate natural language answer
        result_component = GraphResult(
            nodes=graph_result.nodes,
            edges=graph_result.edges,
            paths=graph_result.paths,
            query=query,
            format_style=self._format_style,
        )
        result_text = result_component.format_for_llm().args["result"]

        answer = await explain_query_result(
            self._session,
            query=cypher_string,
            result=result_text,
            original_question=question,
        )
        return answer

    async def _validate_and_repair(
        self, cypher_string: str, schema_text: str
    ) -> str:
        """Validate Cypher syntax; repair via LLM if invalid.

        Attempts up to ``_max_repair_attempts`` repairs.  Returns the last
        generated string whether or not it passed validation, so the caller
        always gets a best-effort answer.

        Args:
            cypher_string: Cypher query to validate.
            schema_text: Formatted schema text used when requesting repairs.

        Returns:
            The validated (or best-effort repaired) Cypher string.
        """
        for attempt in range(self._max_repair_attempts + 1):
            query = CypherQuery(query_string=cypher_string)
            is_valid, error = await self._backend.validate_query(query)

            if is_valid:
                return cypher_string

            if attempt < self._max_repair_attempts:
                improved = await suggest_query_improvement(
                    self._session,
                    query=cypher_string,
                    error_message=error or "Unknown syntax error",
                    schema=schema_text,
                )
                cypher_string = improved.query

        # Return last attempt regardless (best-effort)
        return cypher_string


# ============================================================================
# Layer 1 - KG traversal helpers
# ============================================================================


def _node_to_text(node: Any) -> str:
    """Format a GraphNode as entity text for LLM prompts.

    Produces ``(Label: NAME, desc: "...", props: {...})`` used by the
    @generative alignment / pruning functions.

    Args:
        node: GraphNode instance.

    Returns:
        Formatted entity string.
    """
    name = str(node.properties.get("name", node.id)).strip().upper()
    desc = node.properties.get("description", "")
    _SKIP = {"name", "description", "embedding"}
    props = {
        k: v
        for k, v in node.properties.items()
        if k not in _SKIP and not k.startswith("_")
    }
    parts = [f"({node.label}: {name}"]
    if desc:
        parts.append(f', desc: "{str(desc).replace(chr(34), chr(39))}"')
    if props:
        prop_items = [f"{k}: {v}" for k, v in list(props.items())[:8]]
        parts.append(f", props: {{{', '.join(prop_items)}}}")
    parts.append(")")
    return "".join(parts)


def _edge_to_triplet_text(edge: Any) -> str:
    """Format a GraphEdge as a triplet text for LLM prompts.

    Produces ``(Src)-[REL, props: {...}]->(Tgt)`` format.

    Args:
        edge: GraphEdge instance.

    Returns:
        Formatted triplet string.
    """
    src = _node_to_text(edge.source)
    tgt = _node_to_text(edge.target)
    _SKIP = {"embedding"}
    props = {
        k: v
        for k, v in edge.properties.items()
        if k not in _SKIP and not k.startswith("_")
    }
    if props:
        prop_items = [f"{k}: {v}" for k, v in list(props.items())[:5]]
        return f"{src}-[{edge.label}, props: {{{', '.join(prop_items)}}}]->{tgt}"
    return f"{src}-[{edge.label}]->{tgt}"


async def _search_entities_by_name(
    backend: GraphBackend, name: str, k: int = 4
) -> list:
    """Search KG entities by case-insensitive name containment.

    Args:
        backend: Graph database backend.
        name: Entity name fragment to search for.
        k: Maximum number of results.

    Returns:
        List of matching GraphNode objects.
    """
    q = CypherQuery(
        query_string=(
            "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($name) "
            "RETURN n LIMIT $k"
        ),
        parameters={"name": name, "k": k},
    )
    try:
        result = await backend.execute_query(q)
        return result.nodes
    except Exception:
        return []


async def _search_entities_by_embedding(
    backend: GraphBackend,
    embedding: list,
    k: int = 10,
    exclude_ids: Optional[set] = None,
) -> list:
    """Search KG entities using a Neo4j vector index.

    Falls back gracefully to an empty list when no vector index exists.

    Args:
        backend: Graph database backend.
        embedding: Query embedding vector.
        k: Maximum number of results to return.
        exclude_ids: Node IDs to exclude from the returned list.

    Returns:
        List of GraphNode objects ordered by similarity.
    """
    exclude_ids = exclude_ids or set()
    fetch_k = k + len(exclude_ids)
    q = CypherQuery(
        query_string=(
            "CALL db.index.vector.queryNodes('entity_embedding', $k, $emb) "
            "YIELD node RETURN node"
        ),
        parameters={"k": fetch_k, "emb": embedding},
    )
    try:
        result = await backend.execute_query(q)
        return [n for n in result.nodes if n.id not in exclude_ids][:k]
    except Exception:
        return []


async def _get_unique_relation_types(
    backend: GraphBackend, node_id: str, width: int = 30
) -> list:
    """Retrieve distinct ``(relation_type, target_label)`` pairs from a node.

    Args:
        backend: Graph database backend.
        node_id: Element ID of the source node.
        width: Maximum number of distinct relation types to return.

    Returns:
        List of ``(relation_type, target_label)`` tuples.
    """
    q = CypherQuery(
        query_string=(
            "MATCH (n)-[r]->(m) WHERE elementId(n) = $nid "
            "RETURN DISTINCT type(r) AS rel_type, labels(m)[0] AS tgt_type "
            "LIMIT $w"
        ),
        parameters={"nid": node_id, "w": width},
    )
    try:
        result = await backend.execute_query(q)
        pairs: list = []
        # raw_result holds Neo4j records for non-node/edge RETURN clauses
        if result.raw_result:
            for record in result.raw_result:
                try:
                    data = record.data() if hasattr(record, "data") else dict(record)
                    rt = data.get("rel_type")
                    tt = data.get("tgt_type") or "Unknown"
                    if rt:
                        pairs.append((str(rt), str(tt)))
                except Exception:
                    continue
        # Fallback: deduplicate from edges when the backend already parsed them
        if not pairs and result.edges:
            seen: set = set()
            for edge in result.edges:
                key = (edge.label, edge.target.label)
                if key not in seen:
                    seen.add(key)
                    pairs.append(key)
        return pairs
    except Exception:
        return []


async def _get_triplets(
    backend: GraphBackend,
    node_id: str,
    rel_type: str,
    target_type: str = "Unknown",
    k: int = 30,
) -> list:
    """Retrieve full ``(source)-[rel]->(target)`` triplets from the KG.

    Args:
        backend: Graph database backend.
        node_id: Element ID of the source node.
        rel_type: Relationship type to traverse.
        target_type: Label of target nodes; ignored when ``"Unknown"``/``"None"``.
        k: Maximum number of triplets to return.

    Returns:
        List of GraphEdge objects (each carries source and target GraphNodes).
    """
    # Sanitise identifiers to prevent Cypher injection
    safe_rel = "".join(c for c in rel_type if c.isalnum() or c == "_")
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
    q = CypherQuery(query_string=cypher, parameters={"nid": node_id, "k": k})
    try:
        result = await backend.execute_query(q)
        return result.edges
    except Exception:
        return []


# ============================================================================
# Layer 1 - QA Orchestration (Multi-Route Question Answering)
# ============================================================================


async def orchestrate_qa_retrieval(
    session: Any,
    backend: GraphBackend,
    query: str,
    query_time: str = "",
    domain: str = "general",
    num_routes: int = 3,
    hints: str = "",
    eval_session: Optional[Any] = None,
    emb_client: Optional[Any] = None,
    width: int = 30,
    depth: int = 3,
) -> str:
    """Orchestrate multi-route QA via Think-on-Graph (ToG) algorithm.

    Implements the full Think-on-Graph pipeline:

    1. Break the question into ``num_routes`` solving routes.
    2. In parallel, compute a direct LLM answer (``attempt``) and explore the
       first two routes.
    3. After each new explored route (starting from route 2), call
       ``validate_consensus`` to check whether answers agree.  Stop early on
       consensus.
    4. If consensus is never reached, return the direct answer as fallback.

    Each route exploration performs up to ``depth`` hops of graph traversal:
    extract topic entities → align with KG → prune relations → retrieve
    triplets → prune triplets → evaluate knowledge sufficiency.

    Args:
        session: Mellea session for main LLM calls (question-decomposition,
            entity alignment, relation/triplet pruning).
        backend: Graph database backend used for all Cypher queries.
        query: Natural language question to answer.
        query_time: Temporal context string (e.g. ``"2024-03-05"``).
        domain: Knowledge domain hint (e.g. ``"movie"``).
        num_routes: Number of solving routes to generate and explore.
        hints: Domain-specific text hints appended to prompts.
        eval_session: Separate session for evaluation calls (knowledge
            sufficiency, consensus, direct answer).  Defaults to ``session``.
        emb_client: Optional async OpenAI-compatible embedding client.  When
            provided, entity alignment also uses vector-index search.
        width: Maximum entities / relations considered at each step.
        depth: Maximum graph-traversal hops per route.

    Returns:
        Natural language answer string.
    """
    import asyncio

    _eval = eval_session or session

    # ------------------------------------------------------------------
    # Inner helpers
    # ------------------------------------------------------------------

    async def _embed(texts: list) -> list:
        """Return embeddings via emb_client, or None placeholders."""
        if emb_client is None or not texts:
            return [None] * len(texts)
        try:
            model_name = getattr(emb_client, "_model_name", "text-embedding-3-small")
            response = await emb_client.embeddings.create(
                input=texts, model=model_name
            )
            return [item.embedding for item in response.data]
        except Exception:
            return [None] * len(texts)

    async def _align_topic(route: list, topic_name: str, topic_emb: Any, top_k: int = 45) -> list:
        """Align one topic entity name with KG candidates.

        Returns list of ``(GraphNode, float)`` scored pairs.
        """
        norm_coeff = 1.0 / max(1, len(route))

        fuzzy_nodes = await _search_entities_by_name(backend, topic_name, k=4)
        fuzzy_ids = {n.id for n in fuzzy_nodes}

        emb_nodes: list = []
        if topic_emb is not None and len(fuzzy_nodes) < top_k:
            emb_nodes = await _search_entities_by_embedding(
                backend, topic_emb, k=top_k - len(fuzzy_nodes), exclude_ids=fuzzy_ids
            )

        all_nodes = fuzzy_nodes + emb_nodes
        if not all_nodes:
            return []

        entities_dict = {f"ent_{i}": n for i, n in enumerate(all_nodes)}
        entities_str = "\n".join(
            f"{k}: {_node_to_text(v)}" for k, v in entities_dict.items()
        )

        align_result = await align_topic_entities(
            session,
            query=query,
            query_time=query_time,
            route=route,
            domain=domain,
            top_k_entities_str=entities_str,
        )

        scored = []
        for key, score_str in align_result.relevant_entities.items():
            try:
                score = float(score_str)
            except (TypeError, ValueError):
                score = 0.0
            if score > 0 and key in entities_dict:
                scored.append((entities_dict[key], norm_coeff * score))
        return scored

    async def _explore_one_route(route: list) -> dict:
        """Run the ToG traversal for one solving route.

        Returns dict with keys ``ans``, ``context``, ``route``.
        """
        # Step A: Extract topic entities
        topic_result = await extract_topic_entities(
            session,
            query=query,
            query_time=query_time,
            route=route,
            domain=domain,
        )
        raw_topics = topic_result.entities or []
        topic_names = [str(t).strip().upper() for t in raw_topics if t]

        if not topic_names:
            return {"ans": "I don't know.", "context": "", "route": route}

        # Step B: Align each topic entity with KG
        topic_embeddings = await _embed(topic_names)
        align_tasks = [
            _align_topic(route, name, emb, top_k=min(45, width))
            for name, emb in zip(topic_names, topic_embeddings)
        ]
        align_results = await asyncio.gather(*align_tasks)

        # Aggregate scores across topics (sum for same node)
        score_map: dict = {}
        for scored_list in align_results:
            for node, score in scored_list:
                prev = score_map.get(node.id, (node, 0.0))[1]
                score_map[node.id] = (node, prev + score)

        topic_entities_scores = list(score_map.values())
        initial_entities = [n for n, _ in topic_entities_scores]

        # Step C: Initial knowledge sufficiency check (before any traversal)
        ent_str = (
            "\n".join(f"ent_{i}: {_node_to_text(n)}" for i, n in enumerate(initial_entities))
            or "None"
        )
        eval_result = await evaluate_knowledge_sufficiency(
            _eval,
            query=query,
            query_time=query_time,
            route=route,
            domain=domain,
            entities=ent_str,
            triplets="None",
            hints=hints,
        )
        if eval_result.sufficient.lower().strip() == "yes":
            return {
                "ans": eval_result.answer,
                "context": f"Knowledge Entities:\n{ent_str}\nKnowledge Triplets:\nNone",
                "route": route,
            }

        # Step D: Multi-hop traversal
        cluster_chain: list = []
        visited_edges: set = set()

        for _hop in range(depth):
            triplet_scored: list = []

            for node, entity_score in topic_entities_scores:
                rel_types = await _get_unique_relation_types(backend, node.id, width=width)
                if not rel_types:
                    continue

                entity_str = _node_to_text(node)
                src_name = node.properties.get("name", node.id).strip().upper()
                rels_str = "\n".join(
                    f"rel_{i}: ({node.label}: {src_name})-[{rt}]->({tt}: None)"
                    for i, (rt, tt) in enumerate(rel_types)
                )

                rel_prune = await prune_relations(
                    session,
                    query=query,
                    query_time=query_time,
                    route=route,
                    domain=domain,
                    entity_str=entity_str,
                    relations_str=rels_str,
                    width=width,
                    hints=hints,
                )

                for key, score_str in rel_prune.relevant_relations.items():
                    try:
                        score = float(score_str)
                        idx = int(key.split("_")[1])
                    except (TypeError, ValueError, IndexError):
                        continue
                    if score <= 0 or idx >= len(rel_types):
                        continue
                    rt, tt = rel_types[idx]

                    triplets = await _get_triplets(backend, node.id, rt, tt, k=width)
                    triplets = [e for e in triplets if e.id not in visited_edges]
                    if not triplets:
                        continue

                    trips_str = "\n".join(
                        f"rel_{j}: {_edge_to_triplet_text(e)}"
                        for j, e in enumerate(triplets[:width])
                    )
                    trip_prune = await prune_triplets(
                        session,
                        query=query,
                        query_time=query_time,
                        route=route,
                        domain=domain,
                        entity_str=entity_str,
                        relations_str=trips_str,
                        hints=hints,
                    )

                    for tkey, tscore_str in trip_prune.relevant_relations.items():
                        try:
                            tscore = float(tscore_str)
                            tidx = int(tkey.split("_")[1])
                        except (TypeError, ValueError, IndexError):
                            continue
                        if tscore > 0 and tidx < len(triplets):
                            triplet_scored.append(
                                (triplets[tidx], entity_score * score * tscore)
                            )

            # Keep top-width triplets by score
            triplet_scored.sort(key=lambda x: x[1], reverse=True)
            triplet_scored = triplet_scored[:width]

            if not triplet_scored:
                break

            chain_texts = [_edge_to_triplet_text(e) for e, _ in triplet_scored]
            cluster_chain.append(chain_texts)
            for edge, _ in triplet_scored:
                visited_edges.add(edge.id)

            # Advance topic entities to triplet targets for next hop
            next_scores: dict = {}
            norm_sum = sum(s for _, s in triplet_scored)
            norm = 1.0 / norm_sum if norm_sum > 0 else 1.0
            for edge, score in triplet_scored:
                tgt = edge.target
                prev = next_scores.get(tgt.id, (tgt, 0.0))[1]
                next_scores[tgt.id] = (tgt, prev + score * norm)
            topic_entities_scores = list(next_scores.values())

            # Evaluate sufficiency with accumulated knowledge
            ent_str2 = (
                "\n".join(f"ent_{i}: {_node_to_text(n)}" for i, n in enumerate(initial_entities))
                or "None"
            )
            idx = 0
            trip_parts: list = []
            for chain in cluster_chain:
                for t in chain:
                    trip_parts.append(f"rel_{idx}: {t}")
                    idx += 1
            triplets_str = "\n".join(trip_parts) or "None"

            suf = await evaluate_knowledge_sufficiency(
                _eval,
                query=query,
                query_time=query_time,
                route=route,
                domain=domain,
                entities=ent_str2,
                triplets=triplets_str,
                hints=hints,
            )
            if suf.sufficient.lower().strip() == "yes":
                return {
                    "ans": suf.answer,
                    "context": (
                        f"Knowledge Entities:\n{ent_str2}\n"
                        f"Knowledge Triplets:\n{triplets_str}"
                    ),
                    "route": route,
                }

        # Depth exhausted — force a final answer from accumulated knowledge
        ent_str_f = (
            "\n".join(f"ent_{i}: {_node_to_text(n)}" for i, n in enumerate(initial_entities))
            or "None"
        )
        idx = 0
        trip_parts_f: list = []
        for chain in cluster_chain:
            for t in chain:
                trip_parts_f.append(f"rel_{idx}: {t}")
                idx += 1
        trip_str_f = "\n".join(trip_parts_f) or "None"

        final_suf = await evaluate_knowledge_sufficiency(
            _eval,
            query=query,
            query_time=query_time,
            route=route,
            domain=domain,
            entities=ent_str_f,
            triplets=trip_str_f,
            hints=hints,
        )
        return {
            "ans": final_suf.answer,
            "context": (
                f"Knowledge Entities:\n{ent_str_f}\n"
                f"Knowledge Triplets:\n{trip_str_f}"
            ),
            "route": route,
        }

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------

    # Break question into routes
    routes_result = await break_down_question(
        session,
        query=query,
        query_time=query_time,
        domain=domain,
        route=num_routes,
        hints=hints,
    )
    routes = routes_result.routes or []

    if not routes:
        direct = await generate_direct_answer(
            _eval, query=query, query_time=query_time, domain=domain
        )
        return direct.answer

    # Launch direct answer + first two routes in parallel
    parallel_coros = [
        generate_direct_answer(_eval, query=query, query_time=query_time, domain=domain),
        _explore_one_route(routes[0]),
    ]
    if len(routes) > 1:
        parallel_coros.append(_explore_one_route(routes[1]))

    parallel_results = await asyncio.gather(*parallel_coros)
    direct_result = parallel_results[0]
    route_results = list(parallel_results[1:])
    attempt = f'"{direct_result.answer}". {direct_result.reason}'

    # Explore remaining routes, checking consensus after each
    final = attempt
    stop = False

    for i, route in enumerate(routes[2:], start=2):
        route_results.append(await _explore_one_route(route))

        n_total = len(routes)
        n_explored = len(route_results)
        n_remaining = n_total - n_explored
        routes_info = (
            f"\nWe have identified {n_total} solving route(s) below, "
            f"and have {n_remaining} unexplored solving route(s) left.:\n"
        )
        for j, rr in enumerate(route_results):
            route_label = routes[j] if j < len(routes) else []
            routes_info += (
                f"Route {j + 1}: {route_label}\n"
                f"Reference: {rr['context']}\n"
                f"Answer: {rr['ans']}\n\n"
            )
        for j in range(n_explored, n_total):
            routes_info += f"Route {j + 1}: {routes[j]}\n\n"

        val = await validate_consensus(
            _eval,
            query=query,
            query_time=query_time,
            domain=domain,
            attempt=attempt,
            routes_info=routes_info,
            hints=hints,
        )
        final = val.final_answer
        stop = val.judgement.lower().strip().replace(" ", "") == "yes"
        if stop:
            break

    if not stop and len(route_results) >= 2:
        # Final consensus check using all explored routes
        n_total = len(routes)
        routes_info = (
            f"\nWe have identified {n_total} solving route(s) below, "
            "and have 0 unexplored solving route(s) left.:\n"
        )
        for j, rr in enumerate(route_results):
            route_label = routes[j] if j < len(routes) else []
            routes_info += (
                f"Route {j + 1}: {route_label}\n"
                f"Reference: {rr['context']}\n"
                f"Answer: {rr['ans']}\n\n"
            )
        val = await validate_consensus(
            _eval,
            query=query,
            query_time=query_time,
            domain=domain,
            attempt=attempt,
            routes_info=routes_info,
            hints=hints,
        )
        final = val.final_answer

    return final


# ============================================================================
# Layer 1 - Update Orchestration (Document-based KG Updating)
# ============================================================================


async def orchestrate_kg_update(
    session: MelleaSession,
    backend: GraphBackend,
    doc_text: str,
    domain: str = "general",
    hints: str = "",
    entity_types: str = "",
    relation_types: str = "",
) -> dict:
    """Orchestrate KG update pipeline.

    This is the main Layer 1 entry point for updating a knowledge graph with
    information extracted from documents. It extracts entities and relations,
    aligns them with existing KG data, and decides on merges.

    Args:
        session: Mellea session for LLM calls
        backend: Graph database backend for queries and updates
        doc_text: Document text to extract information from
        domain: Domain-specific knowledge
        hints: Domain-specific hints for the LLM
        entity_types: Comma-separated list of valid entity types
        relation_types: Comma-separated list of valid relation types

    Returns:
        Dictionary with:
        - extracted_entities: List of extracted entity objects
        - extracted_relations: List of extracted relation objects
        - aligned_entities: List of alignment results
        - aligned_relations: List of alignment results
        - update_summary: Summary of updates made to KG
    """
    # Step 1: Extract entities and relations from document
    extraction = await extract_entities_and_relations(
        session,
        doc_context=doc_text,
        domain=domain,
        hints=hints,
        reference="",
        entity_types=entity_types,
        relation_types=relation_types,
    )

    # Step 2-3: Align entities with KG and decide merges
    # (Simplified - full implementation would iterate through extracted entities)

    # Step 4-5: Align relations with KG and decide merges
    # (Simplified - full implementation would iterate through extracted relations)

    return {
        "extracted_entities": extraction.entities,
        "extracted_relations": extraction.relations,
        "aligned_entities": [],
        "aligned_relations": [],
        "update_summary": "Document processed and entities/relations extracted",
    }


__all__ = [
    # Main Layer 1 orchestration functions
    "KGRag",
    "format_schema",
    "orchestrate_qa_retrieval",
    "orchestrate_kg_update",
    # QA Generative functions (Layer 3)
    "break_down_question",
    "extract_topic_entities",
    "align_topic_entities",
    "prune_relations",
    "prune_triplets",
    "evaluate_knowledge_sufficiency",
    "validate_consensus",
    "generate_direct_answer",
    # Update Generative functions (Layer 3)
    "extract_entities_and_relations",
    "align_entity_with_kg",
    "decide_entity_merge",
    "align_relation_with_kg",
    "decide_relation_merge",
]
