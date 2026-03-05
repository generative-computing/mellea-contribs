"""KGRag: Knowledge Graph Retrieval-Augmented Generation.

Layer 1 application that orchestrates the full KG RAG pipeline:
  1. Convert a natural language question to a Cypher query (Layer 3 @generative)
  2. Validate and repair the query if needed (Layer 3 requirements + repair loop)
  3. Execute the validated query against a graph database (Layer 4 backend)
  4. Format the results for LLM consumption (Layer 2 components)
  5. Generate a natural language answer (Layer 3 @generative)

Example::

    import asyncio
    from mellea import start_session
    from mellea_contribs.kg import Neo4jBackend
    from mellea_contribs.kg.kgrag import KGRag

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

from mellea import MelleaSession

from mellea_contribs.kg.components.llm_guided import (
    explain_query_result,
    natural_language_to_cypher,
    suggest_query_improvement,
)
from mellea_contribs.kg.components.query import CypherQuery
from mellea_contribs.kg.components.result import GraphResult
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
