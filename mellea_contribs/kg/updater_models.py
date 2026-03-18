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
        default=32,
        description="Number of documents to process in parallel.",
    )

    merge_strategy: str = Field(
        default="merge_if_similar",
        description="How to handle entity/relation conflicts: "
        "'merge_if_similar' (default), 'skip', 'overwrite', 'create_variant'.",
    )

    similarity_threshold: float = Field(
        default=0.8,
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

    entity_types: Optional[list[str]] = Field(
        default=None,
        description="List of entity types to extract. "
        "If None, extract all types found.",
    )

    relation_types: Optional[list[str]] = Field(
        default=None,
        description="List of relation types to extract. "
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

    alignment_model: str = Field(
        default="gpt-4o-mini",
        description="Model for entity alignment.",
    )

    alignment_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for alignment LLM.",
    )

    merge_decision_model: str = Field(
        default="gpt-4o-mini",
        description="Model for merge decisions.",
    )

    merge_decision_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for merge decision LLM.",
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

    total_documents: int = Field(
        default=0, description="Total documents processed"
    )

    successful_documents: int = Field(
        default=0, description="Documents processed successfully"
    )

    failed_documents: int = Field(
        default=0, description="Documents that failed processing"
    )

    # Extraction statistics
    entities_extracted: int = Field(
        default=0, description="Total entities extracted from documents"
    )

    relations_extracted: int = Field(
        default=0, description="Total relations extracted from documents"
    )

    # Alignment statistics
    entities_aligned: int = Field(
        default=0, description="Entities aligned with existing KG entities"
    )

    entities_new: int = Field(
        default=0, description="New entities added to KG"
    )

    relations_aligned: int = Field(
        default=0, description="Relations aligned with existing KG relations"
    )

    relations_new: int = Field(
        default=0, description="New relations added to KG"
    )

    # Merge statistics
    entities_merged: int = Field(
        default=0, description="Entities merged with existing entities"
    )

    relations_merged: int = Field(
        default=0, description="Relations merged with existing relations"
    )

    entities_skipped: int = Field(
        default=0, description="Entities skipped due to merge conflicts"
    )

    relations_skipped: int = Field(
        default=0, description="Relations skipped due to merge conflicts"
    )

    # Performance
    average_processing_time_per_doc_ms: float = Field(
        default=0.0, description="Average processing time per document (milliseconds)"
    )

    total_processing_time_ms: float = Field(
        default=0.0, description="Total processing time"
    )


class MergeConflict(BaseModel):
    """Record of a merge conflict during KG update.

    Tracks conflicts that occurred when trying to align/merge entities or relations.
    """

    source_id: str = Field(description="ID of source entity/relation")

    target_id: str = Field(description="ID of target entity/relation")

    conflict_type: str = Field(
        description="Type of conflict: 'entity_merge', 'relation_merge', 'property_conflict'"
    )

    similarity_score: float = Field(
        ge=0.0, le=1.0, description="Similarity score between entities/relations"
    )

    decision: str = Field(
        description="How conflict was resolved: 'merged', 'skipped', 'variant_created'"
    )

    reason: str = Field(description="Reasoning for the decision")

    timestamp: Optional[str] = Field(
        default=None, description="When conflict occurred (ISO format)"
    )


class UpdateResult(BaseModel):
    """Result of updating KG with entities and relations from a document.

    Tracks what was added, merged, and skipped during update for a single document.
    """

    document_id: str = Field(description="ID of document being processed")

    success: bool = Field(
        default=True, description="Whether update completed successfully"
    )

    # Extraction results
    entities_found: int = Field(
        default=0, description="Number of entities found in document"
    )

    relations_found: int = Field(
        default=0, description="Number of relations found in document"
    )

    # What was added/merged
    entities_added: int = Field(
        default=0, description="Number of new entities added to KG"
    )

    entities_merged: int = Field(
        default=0, description="Number of entities merged with existing KG entities"
    )

    entities_skipped: int = Field(
        default=0, description="Number of entities skipped due to conflicts"
    )

    relations_added: int = Field(
        default=0, description="Number of new relations added to KG"
    )

    relations_merged: int = Field(
        default=0, description="Number of relations merged with existing KG relations"
    )

    relations_skipped: int = Field(
        default=0, description="Number of relations skipped due to conflicts"
    )

    # Conflicts
    conflicts: list[MergeConflict] = Field(
        default_factory=list, description="List of merge conflicts encountered"
    )

    # Metadata
    processing_time_ms: float = Field(
        default=0.0, description="Time to process document (milliseconds)"
    )

    model_used: str = Field(
        default="gpt-4o-mini", description="Extraction model used"
    )

    error: Optional[str] = Field(default=None, description="Error message if failed")

    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal warnings during processing"
    )


class UpdateBatchResult(BaseModel):
    """Aggregated results from batch KG update.

    Combines results from updating multiple documents with statistics.
    """

    total_documents: int = Field(
        default=0, description="Total documents processed"
    )

    successful_documents: int = Field(
        default=0, description="Number of successful documents"
    )

    failed_documents: int = Field(
        default=0, description="Number of failed documents"
    )

    results: list[UpdateResult] = Field(
        default_factory=list, description="Per-document results"
    )

    stats: UpdateStats = Field(
        default_factory=UpdateStats, description="Aggregated statistics"
    )

    total_time_ms: float = Field(
        default=0.0, description="Total time for batch (milliseconds)"
    )

    avg_time_per_document_ms: float = Field(
        default=0.0, description="Average time per document (milliseconds)"
    )

    start_time: Optional[str] = Field(
        default=None, description="When batch processing started (ISO format)"
    )

    end_time: Optional[str] = Field(
        default=None, description="When batch processing ended (ISO format)"
    )


class KGUpdateRunConfig(BaseModel):
    """Aggregated run-time configuration for the KG update pipeline.

    Collects all settings needed by a KG update run script: session (LLM model),
    updater (concurrency, loop budgets), dataset (path, domain, progress), and
    graph backend (connection, mock flag).

    Typical usage::

        config = KGUpdateRunConfig(
            model="gpt-4o-mini",
            num_workers=32,
            domain="movie",
            dataset_path="data/corpus.jsonl.bz2",
        )
    """

    # Session
    model: str = Field(default="gpt-4o-mini", description="LLM model for extraction and alignment.")

    # Updater concurrency
    num_workers: int = Field(default=64, description="Max concurrent async workers.")
    queue_size: int = Field(default=64, description="Async queue capacity for document loading.")

    # Loop budgets (passed to orchestrate_kg_update)
    extraction_loop_budget: int = Field(
        default=3, description="Retry budget for entity/relation extraction."
    )
    alignment_loop_budget: int = Field(
        default=2, description="Retry budget for entity alignment."
    )
    align_topk: int = Field(
        default=10, description="Number of top candidates considered during alignment."
    )
    align_entity: bool = Field(default=True, description="Whether to align entities.")
    merge_entity: bool = Field(default=True, description="Whether to merge aligned entities.")
    align_relation: bool = Field(default=True, description="Whether to align relations.")
    merge_relation: bool = Field(default=True, description="Whether to merge aligned relations.")

    # Dataset
    dataset_path: Optional[str] = Field(default=None, description="Path to input JSONL dataset.")
    domain: str = Field(default="movie", description="Knowledge domain label.")
    progress_path: str = Field(
        default="results/update_kg_progress.json",
        description="Path to the progress checkpoint file.",
    )

    # Graph backend
    db_uri: str = Field(
        default="bolt://localhost:7687", description="Graph database connection URI."
    )
    db_user: str = Field(default="neo4j", description="Graph database username.")
    db_password: str = Field(default="password", description="Graph database password.")
    mock: bool = Field(default=False, description="Use in-memory mock backend instead of real DB.")

    # Misc
    verbose: bool = Field(default=False, description="Enable verbose logging.")


__all__ = [
    "UpdateConfig",
    "UpdateSessionConfig",
    "UpdateStats",
    "MergeConflict",
    "UpdateResult",
    "UpdateBatchResult",
    "KGUpdateRunConfig",
]
