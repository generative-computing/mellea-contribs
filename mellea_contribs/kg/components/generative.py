"""Generative functions for KG-RAG using Mellea's @generative decorator."""
from typing import List

from mellea import generative

from mellea_contribs.kg.components._prompts import (
    ALIGN_TOPIC_ENTITIES,
    BREAK_DOWN_QUESTION,
    EVALUATE_KNOWLEDGE_SUFFICIENCY,
    EXTRACT_TOPIC_ENTITIES,
    GENERATE_DIRECT_ANSWER,
    PRUNE_RELATIONS,
    PRUNE_TRIPLETS,
    VALIDATE_CONSENSUS,
)
from mellea_contribs.kg.models import (
    AlignmentResult,
    DirectAnswer,
    EvaluationResult,
    ExtractionResult,
    MergeDecision,
    QuestionRoutes,
    RelevantEntities,
    RelevantRelations,
    TopicEntities,
    ValidationResult,
)


def _prompt(p: str):
    """Set the docstring (prompt template) before @generative is applied."""
    def decorator(fn):
        fn.__doc__ = p
        return fn
    return decorator


# QA Generative Functions (Layer 3 LLM Functions)


@generative
@_prompt(BREAK_DOWN_QUESTION)
async def break_down_question(
    query: str,
    query_time: str,
    domain: str,
    route: int,
    hints: str
) -> QuestionRoutes:
    pass


@generative
@_prompt(EXTRACT_TOPIC_ENTITIES)
async def extract_topic_entities(
    query: str,
    query_time: str,
    route: List[str],
    domain: str
) -> TopicEntities:
    pass


@generative
@_prompt(ALIGN_TOPIC_ENTITIES)
async def align_topic_entities(
    query: str,
    query_time: str,
    route: List[str],
    domain: str,
    top_k_entities_str: str
) -> RelevantEntities:
    pass


@generative
@_prompt(PRUNE_RELATIONS)
async def prune_relations(
    query: str,
    query_time: str,
    route: List[str],
    domain: str,
    entity_str: str,
    relations_str: str,
    width: int,
    hints: str
) -> RelevantRelations:
    pass


@generative
@_prompt(PRUNE_TRIPLETS)
async def prune_triplets(
    query: str,
    query_time: str,
    route: List[str],
    domain: str,
    entity_str: str,
    relations_str: str,
    hints: str
) -> RelevantRelations:
    pass


@generative
@_prompt(EVALUATE_KNOWLEDGE_SUFFICIENCY)
async def evaluate_knowledge_sufficiency(
    query: str,
    query_time: str,
    route: List[str],
    domain: str,
    entities: str,
    triplets: str,
    hints: str
) -> EvaluationResult:
    pass


@generative
@_prompt(VALIDATE_CONSENSUS)
async def validate_consensus(
    query: str,
    query_time: str,
    domain: str,
    attempt: str,
    routes_info: str,
    hints: str
) -> ValidationResult:
    pass


@generative
@_prompt(GENERATE_DIRECT_ANSWER)
async def generate_direct_answer(
    query: str,
    query_time: str,
    domain: str
) -> DirectAnswer:
    pass


# Update Generative Functions (will be implemented similarly)
@generative
async def extract_entities_and_relations(
    doc_context: str,
    domain: str,
    hints: str,
    reference: str,
    entity_types: str = "",
    relation_types: str = ""
) -> ExtractionResult:
    """Extract entities and relations from a document context.

    See full docstring in source repository for complete extraction guidelines.
    """
    pass


@generative
async def align_entity_with_kg(
    extracted_entity_name: str,
    extracted_entity_type: str,
    extracted_entity_desc: str,
    candidate_entities: str,
    domain: str,
    doc_text: str = ""
) -> AlignmentResult:
    """You are an expert at aligning extracted entities with an existing knowledge graph in the {domain} domain.

    An entity was extracted from a document:
    - Name: {extracted_entity_name}
    - Type: {extracted_entity_type}
    - Description: {extracted_entity_desc}

    Source document excerpt:
    {doc_text}

    Candidate entities already in the knowledge graph:
    {candidate_entities}

    Your task: determine whether any candidate entity refers to the same real-world entity as the extracted one.
    If a match exists, return its ID and your confidence (0-1). If no match exists, return null for aligned_entity_id.

    Return a JSON object: {{"aligned_entity_id": "<id or null>", "confidence": <0-1>, "reasoning": "<brief reason>"}}
    """
    pass


@generative
async def decide_entity_merge(
    entity_pair: str,
    doc_text: str,
    domain: str
) -> MergeDecision:
    """You are an expert at resolving entity coreference in the {domain} domain.

    Two entities from a knowledge graph may or may not refer to the same real-world entity:
    {entity_pair}

    Supporting document excerpt:
    {doc_text}

    Decide whether these two entities should be merged into one.
    Consider: name similarity, type match, shared properties, and contextual clues.

    Return a JSON object:
    {{"should_merge": true/false, "reasoning": "<brief explanation>", "merged_properties": {{}}}}

    Only populate merged_properties when should_merge is true, combining the best values from both entities.
    """
    pass


@generative
async def align_relation_with_kg(
    extracted_relation: str,
    candidate_relations: str,
    synonym_relations: str,
    domain: str,
    doc_text: str = ""
) -> AlignmentResult:
    """You are an expert at aligning extracted relations with an existing knowledge graph in the {domain} domain.

    An relation was extracted from a document:
    {extracted_relation}

    Source document excerpt:
    {doc_text}

    Candidate relations already in the knowledge graph (same source and target):
    {candidate_relations}

    Known synonym relation types that should be considered equivalent:
    {synonym_relations}

    Your task: determine whether any candidate relation is semantically equivalent to the extracted one.
    If a match exists, return its ID and your confidence (0-1). If no match exists, return null for aligned_entity_id.

    Return a JSON object: {{"aligned_entity_id": "<id or null>", "confidence": <0-1>, "reasoning": "<brief reason>"}}
    """
    pass


@generative
async def decide_relation_merge(
    relation_pair: str,
    doc_text: str,
    domain: str
) -> MergeDecision:
    """You are an expert at resolving relation coreference in the {domain} domain.

    Two relations from a knowledge graph may or may not represent the same real-world fact:
    {relation_pair}

    Supporting document excerpt:
    {doc_text}

    Decide whether these two relations should be merged into one.
    Consider: relation type equivalence, property compatibility, and contextual clues.

    Return a JSON object:
    {{"should_merge": true/false, "reasoning": "<brief explanation>", "merged_properties": {{}}}}

    Only populate merged_properties when should_merge is true, combining the best values from both relations.
    """
    pass
