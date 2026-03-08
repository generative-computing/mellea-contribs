"""Query components for graph database operations."""

from mellea_contribs.kg.components.generative import (
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
    GeneratedQuery,
    explain_query_result,
    natural_language_to_cypher,
    suggest_query_improvement,
)
from mellea_contribs.kg.components.query import CypherQuery, GraphQuery, SparqlQuery
from mellea_contribs.kg.components.result import GraphResult
from mellea_contribs.kg.components.traversal import GraphTraversal

__all__ = [
    "CypherQuery",
    "GeneratedQuery",
    "GraphQuery",
    "GraphResult",
    "GraphTraversal",
    "SparqlQuery",
    "explain_query_result",
    "natural_language_to_cypher",
    "suggest_query_improvement",
    # QA generative functions
    "break_down_question",
    "extract_topic_entities",
    "align_topic_entities",
    "prune_relations",
    "prune_triplets",
    "evaluate_knowledge_sufficiency",
    "validate_consensus",
    "generate_direct_answer",
    # Update generative functions
    "extract_entities_and_relations",
    "align_entity_with_kg",
    "decide_entity_merge",
    "align_relation_with_kg",
    "decide_relation_merge",
]
