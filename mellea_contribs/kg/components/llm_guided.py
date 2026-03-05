"""LLM-guided query construction for knowledge graphs.

Uses Mellea's @generative pattern to convert natural language into graph queries
and to explain/repair query results.
"""

from typing import Any

from mellea import generative
from pydantic import BaseModel


class GeneratedQuery(BaseModel):
    """Pydantic model for a generated graph query."""

    query: str
    explanation: str
    parameters: dict[str, Any] | None = None


@generative
async def natural_language_to_cypher(
    natural_language_query: str,
    graph_schema: str,
    examples: str,
) -> GeneratedQuery:
    """Generate a Cypher query from a natural language question.

    Given a natural language question and the graph schema, generate a
    valid Cypher query that answers the question.

    Graph Schema:
    {graph_schema}

    Examples:
    {examples}

    Question: {natural_language_query}

    Generate a Cypher query to answer this question. Return as JSON:
    {{"query": "MATCH ...", "explanation": "This query...", "parameters": {{}}}}

    Query:"""
    pass


@generative
async def explain_query_result(
    query: str,
    result: str,
    original_question: str,
) -> str:
    """Explain a graph query result in natural language.

    Original Question: {original_question}

    Query Executed:
    {query}

    Results:
    {result}

    Explain what these results mean in relation to the original question.
    Write a clear, natural language answer.

    Answer:"""
    pass


@generative
async def suggest_query_improvement(
    query: str,
    error_message: str,
    schema: str,
) -> GeneratedQuery:
    """Suggest a corrected query based on an error message.

    The following query failed:
    {query}

    Error: {error_message}

    Graph Schema:
    {schema}

    Suggest a corrected version of the query. Return as JSON:
    {{"query": "...", "explanation": "The issue was...", "parameters": {{}}}}

    Corrected Query:"""
    pass
