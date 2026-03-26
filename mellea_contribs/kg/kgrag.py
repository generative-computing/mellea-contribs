"""KGRag: Knowledge Graph Retrieval-Augmented Generation.

Layer 2 library that orchestrates the full KG RAG pipeline:

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
        align_topic_entities,
        break_down_question,
        evaluate_knowledge_sufficiency,
        extract_entities_and_relations,
        extract_topic_entities,
        generate_direct_answer,
        validate_consensus,
    )
    from mellea_contribs.kg.components.llm_guided import (
        explain_query_result,
        natural_language_to_cypher,
    )
    from mellea_contribs.kg.components.retrieval import (
        edge_to_triplet_text,
        fetch_schema_text,
        node_to_text,
        search_and_align_entities,
        traverse_and_prune,
        validate_and_execute_query,
    )
    from mellea_contribs.kg.components.persistence import (
        align_and_upsert_entity,
        align_and_upsert_relation,
    )
    from mellea_contribs.kg.components.result import GraphResult
except ImportError:
    # These are optional - mellea may not be installed
    align_topic_entities = None
    break_down_question = None
    evaluate_knowledge_sufficiency = None
    extract_entities_and_relations = None
    extract_topic_entities = None
    generate_direct_answer = None
    validate_consensus = None
    explain_query_result = None
    natural_language_to_cypher = None
    edge_to_triplet_text = None
    fetch_schema_text = None
    node_to_text = None
    search_and_align_entities = None
    traverse_and_prune = None
    validate_and_execute_query = None
    align_and_upsert_entity = None
    align_and_upsert_relation = None
    GraphResult = None

try:
    from mellea_contribs.kg.graph_dbs.base import GraphBackend
except ImportError:
    GraphBackend = None  # type: ignore[assignment,misc]

try:
    from mellea_contribs.kg.preprocessor import KGPreprocessor
except ImportError:
    KGPreprocessor = None  # type: ignore[assignment,misc]

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
        # Step 1: Get graph schema (Layer 3)
        schema_text = await fetch_schema_text(self._backend)

        # Step 2: Generate Cypher query from natural language (Layer 3 @generative)
        generated = await natural_language_to_cypher(
            self._session,
            natural_language_query=question,
            graph_schema=schema_text,
            examples=examples,
        )
        cypher_string = generated.query

        # Step 3: Validate, repair, and execute (Layer 3)
        graph_result = await validate_and_execute_query(
            self._backend,
            self._session,
            cypher_string,
            schema_text,
            max_repair_attempts=self._max_repair_attempts,
            format_style=self._format_style,
        )

        # Step 4: Format result and generate natural language answer (Layer 3 @generative)
        result_component = GraphResult(
            nodes=graph_result.nodes,
            edges=graph_result.edges,
            paths=graph_result.paths,
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


# ============================================================================
# Layer 2 - QA Orchestration (Multi-Route Question Answering)
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

    async def _explore_one_route(route: list) -> dict:
        """Run the ToG traversal for one solving route.

        Returns dict with keys ``ans``, ``context``, ``route``.
        """
        # Step A: Extract topic entities (Layer 3 @generative)
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

        # Step B: Align each topic entity with KG (Layer 3 executor)
        topic_embeddings = await _embed(topic_names)
        align_tasks = [
            search_and_align_entities(
                backend, session, query, query_time, route, domain,
                name, emb, top_k=min(45, width),
            )
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

        # Step C: Initial knowledge sufficiency check (Layer 3 @generative)
        ent_str = (
            "\n".join(f"ent_{i}: {node_to_text(n)}" for i, n in enumerate(initial_entities))
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
            # One hop via Layer 3 executor (no backend calls here)
            triplet_scored = await traverse_and_prune(
                backend, session, topic_entities_scores, visited_edges,
                query, query_time, route, domain, hints, width,
            )

            # Keep top-width triplets by score
            triplet_scored.sort(key=lambda x: x[1], reverse=True)
            triplet_scored = triplet_scored[:width]

            if not triplet_scored:
                break

            chain_texts = [edge_to_triplet_text(e) for e, _ in triplet_scored]
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

            # Evaluate sufficiency with accumulated knowledge (Layer 3 @generative)
            ent_str2 = (
                "\n".join(f"ent_{i}: {node_to_text(n)}" for i, n in enumerate(initial_entities))
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
            "\n".join(f"ent_{i}: {node_to_text(n)}" for i, n in enumerate(initial_entities))
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
# Layer 2 - Update Orchestration (Document-based KG Updating)
# ============================================================================


async def orchestrate_kg_update(
    preprocessor: "KGPreprocessor",
    backend: GraphBackend,
    doc_text: str,
    align_topk: int = 10,
    confidence_threshold: float = 0.6,
) -> dict:
    """Orchestrate KG update pipeline.

    Uses the provided preprocessor for domain-specific extraction and
    post-processing, then aligns and upserts the results into the KG.

    Pipeline:
      1. Extract entities and relations via the preprocessor (applies domain
         hints and post-processing).
      2. For each extracted entity: search KG candidates, align via LLM,
         decide whether to merge or insert.
      3. Write new / merged entities to the backend.
      4. For each extracted relation: resolve source/target entity IDs, search
         for matching KG relations, align via LLM, decide whether to merge.
      5. Write new / merged relations to the backend.

    Args:
        session: Mellea session for LLM calls.
        backend: Graph database backend for queries and updates.
        doc_text: Document text to extract information from.
        domain: Knowledge domain label (e.g. ``"movie"``).
        hints: Domain-specific hints for the LLM.
        entity_types: Comma-separated list of valid entity types.
        relation_types: Comma-separated list of valid relation types.
        align_topk: Number of KG candidates to retrieve for alignment.
        confidence_threshold: Minimum alignment confidence to trigger a merge.

    Returns:
        Dictionary with:
        - ``extracted_entities``: List of Entity objects from the document.
        - ``extracted_relations``: List of Relation objects from the document.
        - ``aligned_entities``: List of ``(entity, aligned_id | None)`` pairs.
        - ``aligned_relations``: List of ``(relation, aligned_id | None)`` pairs.
        - ``entities_inserted``: Count of new entities added to KG.
        - ``entities_merged``: Count of entities merged with existing KG nodes.
        - ``relations_inserted``: Count of new relations added to KG.
        - ``relations_merged``: Count of relations merged with existing KG edges.
    """
    session = preprocessor.session
    domain = preprocessor.domain

    # ── Step 1: Extract entities and relations via preprocessor ───────────────
    extraction = await preprocessor.process_document(doc_text)

    # ── Step 2-3: Align and upsert entities (Layer 3 executor) ───────────────
    entity_name_to_id: dict = {}
    aligned_entities: list = []
    entities_inserted = 0
    entities_merged = 0

    for entity in extraction.entities:
        entity_id, was_merged = await align_and_upsert_entity(
            backend, session, entity, doc_text, domain,
            align_topk=align_topk,
            confidence_threshold=confidence_threshold,
        )
        entity_name_to_id[entity.name] = entity_id
        aligned_entities.append((entity, entity_id if was_merged else None))
        if was_merged:
            entities_merged += 1
        else:
            entities_inserted += 1

    # ── Step 4-5: Align and upsert relations (Layer 3 executor) ──────────────
    aligned_relations: list = []
    relations_inserted = 0
    relations_merged = 0

    for relation in extraction.relations:
        src_id = entity_name_to_id.get(relation.source_entity)
        tgt_id = entity_name_to_id.get(relation.target_entity)

        if not src_id or not tgt_id:
            aligned_relations.append((relation, None))
            continue

        rel_id, was_merged = await align_and_upsert_relation(
            backend, session, relation, src_id, tgt_id, doc_text, domain,
            align_topk=align_topk,
            confidence_threshold=confidence_threshold,
        )
        aligned_relations.append((relation, rel_id if was_merged else None))
        if was_merged:
            relations_merged += 1
        else:
            relations_inserted += 1

    return {
        "extracted_entities": extraction.entities,
        "extracted_relations": extraction.relations,
        "aligned_entities": aligned_entities,
        "aligned_relations": aligned_relations,
        "entities_inserted": entities_inserted,
        "entities_merged": entities_merged,
        "relations_inserted": relations_inserted,
        "relations_merged": relations_merged,
        "update_summary": (
            f"Inserted {entities_inserted} entities, merged {entities_merged}; "
            f"inserted {relations_inserted} relations, merged {relations_merged}."
        ),
    }


__all__ = [
    # Layer 2 orchestration functions
    "KGRag",
    "format_schema",
    "orchestrate_qa_retrieval",
    "orchestrate_kg_update",
]
