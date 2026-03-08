"""Configuration and result models for KG-RAG QA pipeline.

This module provides Pydantic models for configuring and tracking QA operations,
integrating with the generative functions in components/generative.py and the
KGRag orchestrator in kgrag.py.

The models are designed to be reused across different QA scenarios and to track
metrics and configurations for reproducibility.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

from mellea_contribs.kg.models import (
    DirectAnswer,
    EvaluationResult,
    QuestionRoutes,
    RelevantEntities,
    RelevantRelations,
    TopicEntities,
    ValidationResult,
)


class QAConfig(BaseModel):
    """Configuration for QA pipeline parameters.

    Controls how the QA pipeline breaks down questions and searches the graph.
    These parameters directly map to generative function inputs.
    """

    route_count: int = Field(
        default=3,
        description="Number of different solving routes to generate per question. "
        "Higher values explore more possibilities but increase LLM calls.",
    )

    depth: int = Field(
        default=3,
        description="Maximum depth for relation traversal in graph search. "
        "How many hops deep to search from initial entities.",
    )

    width: int = Field(
        default=5,
        description="Maximum number of entities to consider at each level. "
        "Limits breadth of graph search to avoid combinatorial explosion.",
    )

    query_time: Optional[str] = Field(
        default=None,
        description="Query timestamp (ISO format) for time-aware reasoning. "
        "If None, uses current time.",
    )

    domain: str = Field(
        default="generic",
        description="Domain for the QA task (e.g., 'movies', 'medicine'). "
        "Used for domain-specific extraction hints.",
    )

    use_consensus: bool = Field(
        default=True,
        description="Whether to use consensus validation across multiple routes. "
        "If True, multiple routes are evaluated and results merged.",
    )

    format_style: str = Field(
        default="natural",
        description="How to format query results for the LLM. "
        "Options: 'triplets', 'natural', 'paths', 'structured'.",
    )

    max_repair_attempts: int = Field(
        default=2,
        description="Maximum attempts to repair invalid Cypher queries. "
        "Used in query validation and repair loop.",
    )


class QASessionConfig(BaseModel):
    """Configuration for LLM and evaluation settings in QA session.

    Manages session-level settings for QA operations like model parameters
    and evaluation criteria.
    """

    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model to use for QA (e.g., 'gpt-4o-mini', 'claude-3-sonnet'). "
        "Should be a LiteLLM-compatible model ID.",
    )

    llm_temperature: float = Field(
        default=0.1,
        description="Temperature for LLM generation (0.0-2.0). "
        "Lower = more deterministic, higher = more creative.",
    )

    llm_max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens for LLM responses. "
        "If None, uses model default.",
    )

    use_few_shot_examples: bool = Field(
        default=True,
        description="Whether to include few-shot Cypher examples in prompts.",
    )

    eval_model: Optional[str] = Field(
        default=None,
        description="Optional separate model for evaluation/validation. "
        "If None, uses same as llm_model.",
    )

    evaluation_threshold: float = Field(
        default=0.5,
        description="Confidence threshold for evaluation (0.0-1.0). "
        "Answers below this are marked as uncertain.",
    )


class QADatasetConfig(BaseModel):
    """Configuration for QA dataset and batch processing.

    Specifies dataset paths and batch processing parameters for running
    QA on multiple questions.
    """

    dataset_path: str = Field(
        description="Path to dataset file (JSONL format). "
        "Each line: {\"question\": \"...\", \"expected_answer\": \"...\", ...}",
    )

    batch_size: int = Field(
        default=10,
        description="Number of questions to process in parallel.",
    )

    output_path: Optional[str] = Field(
        default=None,
        description="Path to save QA results (JSONL format). "
        "If None, results not persisted.",
    )

    num_workers: int = Field(
        default=4,
        description="Number of parallel workers for batch processing.",
    )

    shuffle: bool = Field(
        default=False,
        description="Whether to shuffle dataset before processing.",
    )

    max_examples: Optional[int] = Field(
        default=None,
        description="Maximum number of examples to process. "
        "If None, process entire dataset.",
    )

    skip_errors: bool = Field(
        default=True,
        description="Whether to continue processing on errors. "
        "If False, halt on first error.",
    )


class QAResult(BaseModel):
    """Result of a single QA query.

    Combines the input question, the generated answer, and intermediate results
    for tracing and evaluation.
    """

    question: str = Field(description="The input question")

    answer: str = Field(description="The final answer generated by the system")

    confidence: float = Field(
        description="Confidence score for the answer (0.0-1.0)", ge=0.0, le=1.0
    )

    # Intermediate results from QA pipeline
    question_routes: Optional[QuestionRoutes] = Field(
        default=None, description="Breakdown routes generated in step 1"
    )

    topic_entities: Optional[TopicEntities] = Field(
        default=None, description="Topic entities extracted in step 2"
    )

    relevant_entities: Optional[RelevantEntities] = Field(
        default=None, description="Relevant entities found in graph"
    )

    relevant_relations: Optional[RelevantRelations] = Field(
        default=None, description="Relevant relations found in graph"
    )

    evaluation_result: Optional[EvaluationResult] = Field(
        default=None, description="Knowledge sufficiency evaluation from step 4"
    )

    validation_result: Optional[ValidationResult] = Field(
        default=None, description="Consensus validation result (if multi-route)"
    )

    direct_answer: Optional[DirectAnswer] = Field(
        default=None, description="Direct answer from LLM (if knowledge insufficient)"
    )

    # Evidence and reasoning
    cypher_query: Optional[str] = Field(
        default=None, description="Cypher query used to search the graph"
    )

    graph_evidence: Optional[dict[str, Any]] = Field(
        default=None, description="Graph data that supported the answer"
    )

    reasoning: str = Field(
        default="", description="Detailed reasoning/explanation for the answer"
    )

    # Metadata
    processing_time_ms: float = Field(
        description="Time taken to process the question (milliseconds)"
    )

    model_used: str = Field(description="LLM model used for this result")

    route_used: Optional[int] = Field(
        default=None, description="Index of route that was used (if multi-route)"
    )

    error: Optional[str] = Field(
        default=None, description="Error message if processing failed"
    )


class QAStats(BaseModel):
    """Statistics from batch QA processing.

    Aggregates metrics across multiple QA queries for performance analysis.
    """

    total_questions: int = Field(description="Total questions processed")

    successful_answers: int = Field(description="Number of successful answers")

    failed_answers: int = Field(description="Number of failed answers")

    average_confidence: float = Field(description="Average confidence score")

    average_processing_time_ms: float = Field(description="Average processing time")

    min_processing_time_ms: float = Field(description="Minimum processing time")

    max_processing_time_ms: float = Field(description="Maximum processing time")

    models_used: list[str] = Field(description="List of models used in processing")

    # Evaluation metrics (if ground truth available)
    exact_match_count: Optional[int] = Field(
        default=None, description="Number of exact match answers"
    )

    partial_match_count: Optional[int] = Field(
        default=None, description="Number of partial match answers"
    )

    no_match_count: Optional[int] = Field(
        default=None, description="Number of non-matching answers"
    )

    mean_reciprocal_rank: Optional[float] = Field(
        default=None, description="MRR metric if ranking-based evaluation"
    )


__all__ = [
    "QAConfig",
    "QASessionConfig",
    "QADatasetConfig",
    "QAResult",
    "QAStats",
]
