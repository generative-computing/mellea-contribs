"""Knowledge Graph library for mellea-contribs.

Backend-agnostic graph database abstraction for graph-based RAG applications.

Optional Dependencies:
    Neo4j support requires: pip install mellea-contribs[kg]

Example:
    Basic usage with MockGraphBackend::

        from mellea_contribs.kg import MockGraphBackend, GraphNode, Entity

        backend = MockGraphBackend()
        node = GraphNode(id="1", label="Person", properties={"name": "Alice"})

        # Create a generic entity
        entity = Entity(
            type="Person",
            name="Alice",
            description="Example person",
            paragraph_start="Alice is",
            paragraph_end="here.",
        )

    With domain-specific entities (movie example)::

        from docs.examples.kgrag.models import MovieEntity

        movie = MovieEntity(
            type="Movie",
            name="Oppenheimer",
            description="2023 film",
            paragraph_start="Oppenheimer is",
            paragraph_end="by Nolan.",
            release_year=2023,
            director="Christopher Nolan"
        )

    With Neo4j::

        from mellea_contribs.kg import Neo4jBackend

        backend = Neo4jBackend(
            uri="bolt://localhost:7687",
            auth=("neo4j", "password")
        )
"""

from typing import Any

from mellea_contribs.kg.base import GraphEdge, GraphNode, GraphPath
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
from mellea_contribs.kg.graph_dbs.base import GraphBackend
from mellea_contribs.kg.graph_dbs.mock import MockGraphBackend
from mellea_contribs.kg.embedder import KGEmbedder
from mellea_contribs.kg.kgrag import KGRag, format_schema
from mellea_contribs.kg.preprocessor import KGPreprocessor
from mellea_contribs.kg.embed_models import (
    EmbeddingConfig,
    EmbeddingResult,
    EmbeddingSimilarity,
    EmbeddingStats,
)
from mellea_contribs.kg.models import (
    DirectAnswer,
    Entity,
    EvaluationResult,
    ExtractionResult,
    QuestionRoutes,
    Relation,
    RelevantEntities,
    RelevantRelations,
    TopicEntities,
    ValidationResult,
)

_neo4j_import_error: Exception | None = None
_Neo4jBackend: Any = None

try:
    from mellea_contribs.kg.graph_dbs.neo4j import Neo4jBackend as _Neo4jBackend
except ImportError as e:
    _neo4j_import_error = e

    def _Neo4jBackend(*args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        """Raise ImportError with helpful message when Neo4j not installed."""
        raise ImportError(
            "Neo4j support requires additional dependencies. "
            "Install with: pip install mellea-contribs[kg]"
        ) from _neo4j_import_error


Neo4jBackend = _Neo4jBackend

__all__ = [
    # Core data structures
    "GraphBackend",
    "GraphEdge",
    "GraphNode",
    "GraphPath",
    # Models - QA/extraction outputs
    "DirectAnswer",
    "EvaluationResult",
    "ExtractionResult",
    "QuestionRoutes",
    "RelevantEntities",
    "RelevantRelations",
    "TopicEntities",
    "ValidationResult",
    # Models - Stored entities/relations (base classes)
    "Entity",
    "Relation",
    # Models - Embedding configuration and results
    "EmbeddingConfig",
    "EmbeddingResult",
    "EmbeddingSimilarity",
    "EmbeddingStats",
    # Layer 1 Applications
    "KGRag",
    "KGPreprocessor",
    "KGEmbedder",
    # Backends (Layer 4)
    "MockGraphBackend",
    "Neo4jBackend",
    # Generative functions - QA
    "break_down_question",
    "extract_topic_entities",
    "align_topic_entities",
    "prune_relations",
    "prune_triplets",
    "evaluate_knowledge_sufficiency",
    "validate_consensus",
    "generate_direct_answer",
    # Generative functions - Update
    "extract_entities_and_relations",
    "align_entity_with_kg",
    "decide_entity_merge",
    "align_relation_with_kg",
    "decide_relation_merge",
    # Utilities
    "format_schema",
]
