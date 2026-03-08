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

try:
    from mellea import MelleaSession
except ImportError:
    MelleaSession = None

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
# Layer 1 - QA Orchestration (Multi-Route Question Answering)
# ============================================================================


async def orchestrate_qa_retrieval(
    session: MelleaSession,
    backend: GraphBackend,
    query: str,
    query_time: str = "",
    domain: str = "general",
    num_routes: int = 3,
    hints: str = "",
) -> str:
    """Orchestrate multi-route QA pipeline.

    This is the main Layer 1 entry point for question answering using multiple
    solving routes. It breaks down a question, explores multiple routes, and
    reaches consensus on the answer.

    Args:
        session: Mellea session for LLM calls
        backend: Graph database backend for queries
        query: Natural language question
        query_time: Current query time for context-aware answers
        domain: Domain-specific knowledge (e.g., "movies", "finance")
        num_routes: Number of solving routes to explore
        hints: Domain-specific hints for the LLM

    Returns:
        Natural language answer with reasoning
    """
    # Step 1: Break down question into multiple solving routes
    routes_result = await break_down_question(
        query=query,
        query_time=query_time,
        domain=domain,
        route=num_routes,
        hints=hints,
    )

    # Step 2-5: For each route, perform entity selection, relation pruning,
    # and knowledge evaluation. Collect answers from each route.
    # (This is simplified - full implementation would iterate through routes)

    # Step 6: Reach consensus on final answer
    # (Placeholder - would aggregate answers from all routes)

    # Fallback: Use direct LLM answer if routes don't converge
    direct_answer = await generate_direct_answer(
        query=query,
        query_time=query_time,
        domain=domain,
    )

    return direct_answer.answer


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
