"""Configuration and result models for KG update pipeline.

This module provides Pydantic models for configuring KG update operations,
integrating with the extraction and alignment functions in components/generative.py
and the KGPreprocessor orchestrator.

The models track configuration for extracting entities/relations from documents,
aligning them with existing KG entities, and updating the graph accordingly.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class UpdateConfig(BaseModel):
    """Configuration for KG update process parameters.

    Controls how entities and relations are extracted, aligned, and merged.
    """

    batch_size: int = Field(
        default=10,
        description="Number of documents to process in parallel.",
    )

    merge_strategy: str = Field(
        default="merge_if_similar",
        description="How to handle entity/relation conflicts: "
        "'merge_if_similar' (default), 'skip', 'overwrite', 'create_variant'.",
    )

    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for merging entities/relations (0.0-1.0). "
        "Only merge if alignment confidence > threshold.",
    )

    max_entities_per_doc: Optional[int] = Field(
        default=None,
        description="Maximum entities to extract per document. "
        "If None, extract all found.",
    )

    max_relations_per_doc: Optional[int] = Field(
        default=None,
        description="Maximum relations to extract per document. "
        "If None, extract all found.",
    )

    domain: str = Field(
        default="generic",
        description="Domain for entity/relation extraction (e.g., 'movies'). "
        "Used for domain-specific extraction hints.",
    )

    entity_types: Optional[str] = Field(
        default=None,
        description="Comma-separated list of entity types to extract. "
        "If None, extract all types found.",
    )

    relation_types: Optional[str] = Field(
        default=None,
        description="Comma-separated list of relation types to extract. "
        "If None, extract all types found.",
    )

    skip_validation: bool = Field(
        default=False,
        description="Whether to skip KG schema validation. "
        "If False, validate extracted entities/relations against schema.",
    )


class UpdateSessionConfig(BaseModel):
    """Configuration for LLM and alignment settings in update session.

    Manages session-level settings for entity/relation extraction and alignment.
    """

    extraction_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model to use for entity/relation extraction.",
    )

    extraction_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for extraction LLM (0.0-2.0).",
    )

    alignment_model: Optional[str] = Field(
        default=None,
        description="Optional separate model for entity alignment. "
        "If None, uses extraction_model.",
    )

    alignment_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for alignment LLM.",
    )

    merge_decision_model: Optional[str] = Field(
        default=None,
        description="Optional separate model for merge decisions. "
        "If None, uses extraction_model.",
    )

    use_few_shot_examples: bool = Field(
        default=True,
        description="Whether to include few-shot examples in prompts.",
    )

    num_alignment_candidates: int = Field(
        default=5,
        description="Number of candidate entities to consider for alignment.",
    )


class UpdateStats(BaseModel):
    """Statistics tracking for KG update process.

    Tracks metrics about entities/relations extracted, merged, and added.
    """

    total_documents: int = Field(description="Total documents processed")

    successful_documents: int = Field(description="Documents processed successfully")

    failed_documents: int = Field(description="Documents that failed processing")

    # Extraction statistics
    total_entities_extracted: int = Field(description="Total entities extracted from documents")

    total_relations_extracted: int = Field(description="Total relations extracted from documents")

    # Alignment statistics
    entities_aligned: int = Field(
        description="Entities aligned with existing KG entities"
    )

    entities_new: int = Field(description="New entities added to KG")

    relations_aligned: int = Field(
        description="Relations aligned with existing KG relations"
    )

    relations_new: int = Field(description="New relations added to KG")

    # Merge statistics
    entities_merged: int = Field(description="Entities merged with existing entities")

    relations_merged: int = Field(description="Relations merged with existing relations")

    entities_skipped: int = Field(
        description="Entities skipped due to merge conflicts"
    )

    relations_skipped: int = Field(
        description="Relations skipped due to merge conflicts"
    )

    # Performance
    average_processing_time_per_doc_ms: float = Field(
        description="Average processing time per document (milliseconds)"
    )

    total_processing_time_ms: float = Field(description="Total processing time")


class MergeConflict(BaseModel):
    """Record of a merge conflict during KG update.

    Tracks conflicts that occurred when trying to align/merge entities or relations.
    """

    conflict_type: str = Field(
        description="Type of conflict: 'entity_merge', 'relation_merge', 'property_conflict'"
    )

    existing_id: str = Field(description="ID of existing entity/relation in KG")

    existing_name: str = Field(description="Name of existing entity/relation")

    extracted_name: str = Field(description="Name of extracted entity/relation")

    similarity_score: float = Field(
        ge=0.0, le=1.0, description="Similarity score between entities/relations"
    )

    decision: str = Field(
        description="How conflict was resolved: 'merged', 'skipped', 'variant_created'"
    )

    reason: str = Field(description="Reasoning for the decision")

    timestamp: str = Field(description="When conflict occurred (ISO format)")


class UpdateResult(BaseModel):
    """Result of updating KG with entities and relations from a document.

    Tracks what was added, merged, and skipped during update for a single document.
    """

    document_id: str = Field(description="ID of document being processed")

    success: bool = Field(description="Whether update completed successfully")

    # Extraction results
    entities_found: int = Field(description="Number of entities found in document")

    relations_found: int = Field(description="Number of relations found in document")

    # What was added/merged
    entities_added: list[str] = Field(
        default_factory=list, description="IDs of new entities added to KG"
    )

    entities_merged: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Pairs of (extracted_id, merged_with_kg_id) for merged entities",
    )

    relations_added: list[str] = Field(
        default_factory=list, description="IDs of new relations added to KG"
    )

    relations_merged: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Pairs of (extracted_id, merged_with_kg_id) for merged relations",
    )

    # Conflicts
    conflicts: list[MergeConflict] = Field(
        default_factory=list, description="List of merge conflicts encountered"
    )

    # Metadata
    processing_time_ms: float = Field(description="Time to process document (milliseconds)")

    model_used: str = Field(description="Extraction model used")

    error: Optional[str] = Field(default=None, description="Error message if failed")

    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal warnings during processing"
    )


class UpdateBatchResult(BaseModel):
    """Aggregated results from batch KG update.

    Combines results from updating multiple documents with statistics.
    """

    results: list[UpdateResult] = Field(description="Per-document results")

    stats: UpdateStats = Field(description="Aggregated statistics")

    start_time: str = Field(description="When batch processing started (ISO format)")

    end_time: str = Field(description="When batch processing ended (ISO format)")

    total_time_ms: float = Field(description="Total time for batch (milliseconds)")

    successful_count: int = Field(description="Number of successful documents")

    failed_count: int = Field(description="Number of failed documents")


__all__ = [
    "UpdateConfig",
    "UpdateSessionConfig",
    "UpdateStats",
    "MergeConflict",
    "UpdateResult",
    "UpdateBatchResult",
]
