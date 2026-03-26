"""Layer 3 retrieval executor functions for KG-RAG.

These executor functions combine Layer 4 backend queries with Layer 3
@generative LLM calls. They form the bridge between Layer 2 orchestrators
and Layer 4 backends, so that Layer 2 never touches the backend directly.
"""

from typing import Any

from mellea_contribs.kg.base import GraphEdge, GraphNode
from mellea_contribs.kg.components.generative import (
    align_topic_entities,
    prune_relations,
    prune_triplets,
)
from mellea_contribs.kg.components.llm_guided import suggest_query_improvement

try:
    from mellea_contribs.kg.graph_dbs.base import GraphBackend
except ImportError:
    GraphBackend = None  # type: ignore[assignment,misc]
    suggest_query_improvement = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Text-formatting helpers (used by both retrieval executors and Layer 2)
# ---------------------------------------------------------------------------


def node_to_text(node: GraphNode) -> str:
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


def edge_to_triplet_text(edge: GraphEdge) -> str:
    """Format a GraphEdge as a triplet text for LLM prompts.

    Produces ``(Src)-[REL, props: {...}]->(Tgt)`` format.

    Args:
        edge: GraphEdge instance.

    Returns:
        Formatted triplet string.
    """
    src = node_to_text(edge.source)
    tgt = node_to_text(edge.target)
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


# ---------------------------------------------------------------------------
# Retrieval executor functions (Layer 3)
# ---------------------------------------------------------------------------


async def search_and_align_entities(
    backend: "GraphBackend",
    session: Any,
    query: str,
    query_time: str,
    route: list[str],
    domain: str,
    topic_name: str,
    topic_embedding: Any,
    top_k: int = 45,
) -> list[tuple[GraphNode, float]]:
    """Search KG for candidates then LLM-score their relevance.

    Combines:
    - ``backend.search_entities_by_name()`` (Layer 4 fuzzy search)
    - ``backend.search_entities_by_embedding()`` (Layer 4 vector search)
    - ``align_topic_entities()`` (Layer 3 @generative LLM scoring)

    Args:
        backend: Graph database backend.
        session: Mellea session for LLM calls.
        query: Natural language question.
        query_time: Temporal context string.
        route: Current solving route (list of sub-objectives).
        domain: Knowledge domain hint.
        topic_name: Name of the topic entity to search for.
        topic_embedding: Optional pre-computed embedding for vector search.
        top_k: Maximum candidate entities to retrieve.

    Returns:
        List of ``(GraphNode, score)`` pairs ordered by relevance, where
        scores are normalized by ``1 / len(route)``.
    """
    norm_coeff = 1.0 / max(1, len(route))

    fuzzy_nodes = await backend.search_entities_by_name(topic_name, k=4)
    fuzzy_ids = {n.id for n in fuzzy_nodes}

    emb_nodes: list[GraphNode] = []
    if topic_embedding is not None and len(fuzzy_nodes) < top_k:
        emb_nodes = await backend.search_entities_by_embedding(
            topic_embedding,
            k=top_k - len(fuzzy_nodes),
            exclude_ids=fuzzy_ids,
        )

    all_nodes = fuzzy_nodes + emb_nodes
    if not all_nodes:
        return []

    entities_dict = {f"ent_{i}": n for i, n in enumerate(all_nodes)}
    entities_str = "\n".join(
        f"{k}: {node_to_text(v)}" for k, v in entities_dict.items()
    )

    align_result = await align_topic_entities(
        session,
        query=query,
        query_time=query_time,
        route=route,
        domain=domain,
        top_k_entities_str=entities_str,
    )

    scored: list[tuple[GraphNode, float]] = []
    for key, score_str in align_result.relevant_entities.items():
        try:
            score = float(score_str)
        except (TypeError, ValueError):
            score = 0.0
        if score > 0 and key in entities_dict:
            scored.append((entities_dict[key], norm_coeff * score))
    return scored


async def traverse_and_prune(
    backend: "GraphBackend",
    session: Any,
    topic_entities_scores: list[tuple[GraphNode, float]],
    visited_edges: set[str],
    query: str,
    query_time: str,
    route: list[str],
    domain: str,
    hints: str,
    width: int,
) -> list[tuple[GraphEdge, float]]:
    """Perform one hop of graph traversal with LLM-guided pruning.

    For each entity in ``topic_entities_scores``:
    1. Fetch available relation types (Layer 4).
    2. LLM-prune to at most ``width`` relevant relations (Layer 3 @generative).
    3. Fetch triplets for each relevant relation (Layer 4).
    4. LLM-prune triplets for relevance (Layer 3 @generative).

    Combines:
    - ``backend.get_relation_types()`` (Layer 4)
    - ``prune_relations()`` (Layer 3 @generative)
    - ``backend.get_triplets()`` (Layer 4)
    - ``prune_triplets()`` (Layer 3 @generative)

    Args:
        backend: Graph database backend.
        session: Mellea session for LLM calls.
        topic_entities_scores: List of ``(GraphNode, entity_score)`` pairs
            representing the frontier entities.
        visited_edges: Set of edge IDs already included in previous hops.
        query: Natural language question.
        query_time: Temporal context string.
        route: Current solving route.
        domain: Knowledge domain hint.
        hints: Domain-specific text hints for LLM prompts.
        width: Maximum relations/triplets to consider at each entity.

    Returns:
        List of ``(GraphEdge, score)`` pairs for new triplets discovered in
        this hop, scored by entity_score × relation_score × triplet_score.
    """
    triplet_scored: list[tuple[GraphEdge, float]] = []

    for node, entity_score in topic_entities_scores:
        rel_types = await backend.get_relation_types(node.id, width=width)
        if not rel_types:
            continue

        entity_str = node_to_text(node)
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

            triplets = await backend.get_triplets(node.id, rt, tt, k=width)
            triplets = [e for e in triplets if e.id not in visited_edges]
            if not triplets:
                continue

            trips_str = "\n".join(
                f"rel_{j}: {edge_to_triplet_text(e)}"
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

    return triplet_scored


# ---------------------------------------------------------------------------
# Schema + query execution helpers (Layer 3) — used by KGRag
# ---------------------------------------------------------------------------


async def fetch_schema_text(backend: "GraphBackend") -> str:
    """Retrieve the graph schema and format it as a human-readable string.

    Wraps ``backend.get_schema()`` (Layer 4) and converts the result into
    a concise text description suitable for LLM prompts.

    Args:
        backend: Graph database backend.

    Returns:
        Formatted schema string, e.g.::

            Graph Schema:
              Node labels: Movie, Person, Award
              Relationship types: ACTED_IN, DIRECTED_BY
              Property keys: name, description, year
    """
    schema = await backend.get_schema()
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


async def validate_and_execute_query(
    backend: "GraphBackend",
    session: Any,
    cypher_string: str,
    schema_text: str,
    max_repair_attempts: int = 2,
    format_style: str = "natural",
) -> Any:
    """Validate a Cypher query, repair if needed, then execute it.

    Combines:
    - ``backend.validate_query()`` (Layer 4)
    - ``suggest_query_improvement()`` (Layer 3 @generative)
    - ``backend.execute_query()`` (Layer 4)

    Attempts up to ``max_repair_attempts`` LLM-guided repairs when the
    query fails validation.  Returns the result from the last successful
    execution (or best-effort if all repairs fail).

    Args:
        backend: Graph database backend.
        session: Mellea session for LLM repair calls.
        cypher_string: Cypher query string to validate and execute.
        schema_text: Formatted schema text used for repair prompts.
        max_repair_attempts: Maximum number of repair attempts before
            proceeding with the last generated query.
        format_style: Result format style passed to the backend.

    Returns:
        ``GraphResult`` Component from the backend execution.
    """
    from mellea_contribs.kg.components.query import CypherQuery

    for attempt in range(max_repair_attempts + 1):
        query = CypherQuery(query_string=cypher_string)
        is_valid, error = await backend.validate_query(query)

        if is_valid:
            break

        if attempt < max_repair_attempts and suggest_query_improvement is not None:
            improved = await suggest_query_improvement(
                session,
                query=cypher_string,
                error_message=error or "Unknown syntax error",
                schema=schema_text,
            )
            cypher_string = improved.query

    query = CypherQuery(query_string=cypher_string)
    return await backend.execute_query(query, format_style=format_style)
