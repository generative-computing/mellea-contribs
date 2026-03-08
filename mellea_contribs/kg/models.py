"""Pydantic models for KG-RAG structured outputs."""
from datetime import datetime
from typing import List, Dict, Optional

from pydantic import BaseModel, Field


# QA Models
class QuestionRoutes(BaseModel):
    """Routes for breaking down a complex question into sub-objectives."""

    reason: str = Field(description="Reasoning for the route ordering")
    routes: List[List[str]] = Field(
        description="List of solving routes, each containing sub-objectives"
    )


class TopicEntities(BaseModel):
    """Extracted topic entities from a query."""

    entities: List[str] = Field(description="List of extracted entity names")


class RelevantEntities(BaseModel):
    """Relevant entities with their scores."""

    reason: str = Field(description="Reasoning for entity relevance")
    relevant_entities: Dict[str, float] = Field(
        description="Mapping of entity index (e.g., 'ent_0') to relevance score"
    )


class RelevantRelations(BaseModel):
    """Relevant relations with their scores."""

    reason: str = Field(description="Reasoning for relation relevance")
    relevant_relations: Dict[str, float] = Field(
        description="Mapping of relation index (e.g., 'rel_0') to relevance score"
    )


class EvaluationResult(BaseModel):
    """Evaluation result for whether knowledge is sufficient to answer."""

    sufficient: str = Field(description="'Yes' or 'No' indicating if knowledge is sufficient")
    reason: str = Field(description="Reasoning for the sufficiency judgment")
    answer: str = Field(description="The answer if sufficient, 'I don't know' otherwise")


class ValidationResult(BaseModel):
    """Validation result for consensus among multiple routes."""

    judgement: str = Field(
        description="'Yes' or 'No' for whether consensus is reached"
    )
    final_answer: str = Field(description="The final answer with explanation")


class DirectAnswer(BaseModel):
    """Direct answer without knowledge graph."""

    sufficient: str = Field(
        description="'Yes' or 'No' indicating if LLM knowledge is sufficient"
    )
    reason: str = Field(description="Reasoning for the answer")
    answer: str = Field(description="The answer or 'I don't know'")


# Update Models
class ExtractedEntity(BaseModel):
    """Extracted entity from document."""

    type: str = Field(description="Entity type (e.g., Person, Movie, Organization)")
    name: str = Field(description="Entity name")
    description: str = Field(description="Brief description of the entity")
    paragraph_start: str = Field(
        description="First 5-30 chars of supporting paragraph"
    )
    paragraph_end: str = Field(description="Last 5-30 chars of supporting paragraph")
    properties: Dict[str, any] = Field(
        default_factory=dict, description="Additional properties"
    )


class ExtractedRelation(BaseModel):
    """Extracted relation between entities."""

    source_entity: str = Field(description="Source entity name")
    relation_type: str = Field(description="Relation type (e.g., acted_in, directed)")
    target_entity: str = Field(description="Target entity name")
    description: str = Field(description="Description of the relation")
    paragraph_start: str = Field(
        description="First 5-30 chars of supporting paragraph"
    )
    paragraph_end: str = Field(description="Last 5-30 chars of supporting paragraph")
    properties: Dict[str, any] = Field(
        default_factory=dict, description="Additional properties"
    )


class ExtractionResult(BaseModel):
    """Result of entity and relation extraction."""

    entities: List[ExtractedEntity] = Field(description="List of extracted entities")
    relations: List[ExtractedRelation] = Field(
        description="List of extracted relations"
    )
    reasoning: str = Field(description="Reasoning for the extractions")


class AlignmentResult(BaseModel):
    """Result of entity alignment with existing KG."""

    aligned_entity_id: Optional[str] = Field(
        description="ID of matched entity in KG, or None"
    )
    confidence: float = Field(description="Confidence score 0-1 for the alignment")
    reasoning: str = Field(description="Reasoning for the alignment decision")


class MergeDecision(BaseModel):
    """Decision on whether to merge entities."""

    should_merge: bool = Field(description="Whether entities should be merged")
    reasoning: str = Field(description="Reasoning for the merge decision")
    merged_properties: Dict[str, any] = Field(
        default_factory=dict, description="Properties of merged entity if merging"
    )
